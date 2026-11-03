"""Chronological FEDformer and twelve-stage physical-residual training."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
import time

_ZONE = timezone(timedelta(hours=8))
_STEP = timedelta(seconds=300)


def build_windows(buckets):
    """Create past-only 90-row histories with twelve complete future labels."""
    result = []
    stamps = [datetime.fromisoformat(row["event_time"]) for row in buckets]
    for index in range(90, len(buckets) - 11):
        window = stamps[index-90:index+12]
        if any(b-a != _STEP for a,b in zip(window, window[1:])):
            continue
        history = buckets[index-90:index]
        target = buckets[index:index+12]
        result.append({"origin": stamps[index], "target_end": datetime.fromisoformat(target[-1]["bucket_end"]),
                       "times": [row["event_time"] for row in history],
                       "values": [row["values"] for row in history],
                       "targets": [row["values"][-1] for row in target]})
    return result


def select_split(samples, start, end):
    return [sample for sample in samples if sample["origin"] >= start and sample["target_end"] <= end]


def fit_scaler(buckets, cutoff):
    rows = [row["values"] for row in buckets if datetime.fromisoformat(row["bucket_end"]) <= cutoff]
    if not rows or any(len(row) != 6 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not isfinite(v) for v in row) for row in rows):
        raise ValueError("训练标准化输入为空或含非法数值")
    means = [sum(row[col] for row in rows)/len(rows) for col in range(6)]
    scales = [sqrt(sum((row[col]-means[col])**2 for row in rows)/len(rows)) or 1.0 for col in range(6)]
    return means, scales


def time_marks(stamps):
    result = []
    for stamp in stamps:
        local = (stamp + timedelta(seconds=240)).astimezone(_ZONE)
        result.append([local.hour/23-.5, local.weekday()/6-.5, (local.day-1)/30-.5,
                       (local.timetuple().tm_yday-1)/365-.5])
    return result


def forecast_metrics(truth, predictions, threshold=.65):
    if not truth or len(truth) != len(predictions) or not truth[0] or any(len(a)!=len(truth[0]) or len(a)!=len(b) for a,b in zip(truth,predictions)):
        raise ValueError("评估序列维度不一致")
    if any(not isfinite(value) for matrix in (truth,predictions) for row in matrix for value in row):
        raise ValueError("评估序列含非有限值")
    def detection(pairs):
        counts = dict(tp=0,fp=0,tn=0,fn=0)
        for actual,predicted in pairs:
            counts["tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"] += 1
        return counts
    errors = [b-a for actual,predicted in zip(truth,predictions) for a,b in zip(actual,predicted)]
    result = dict(interval_pairs=len(errors),window_pairs=len(truth),
                  mae=sum(abs(e) for e in errors)/len(errors),rmse=sqrt(sum(e*e for e in errors)/len(errors)),
                  bias=sum(errors)/len(errors),threshold=threshold,
                  interval_detection=detection((a>=threshold,b>=threshold) for x,y in zip(truth,predictions) for a,b in zip(x,y)),
                  window_detection=detection((max(a)>=threshold,max(b)>=threshold) for a,b in zip(truth,predictions)),
                  negative_predictions=sum(v<0 for row in predictions for v in row),per_lead=[])
    for col in range(len(truth[0])):
        error = [predicted[col]-actual[col] for actual,predicted in zip(truth,predictions)]
        result["per_lead"].append(dict(lead_seconds=(col+1)*300,mae=sum(abs(v) for v in error)/len(error),
                                      rmse=sqrt(sum(v*v for v in error)/len(error))))
    return result


def validation_assessment(evaluation):
    models = evaluation["residual_validation"]
    baseline,professional = models["persistence"],models["residual_corrected"]
    improved = professional["mae"] < baseline["mae"]
    return dict(dataset="January residual_validation",criterion="concentration_mae",
                default_candidate="residual_corrected" if improved else "persistence",
                professional_mae_improved=improved,
                persistence_mae=baseline["mae"],professional_mae=professional["mae"],
                warning_tradeoff={"persistence":dict(baseline["window_detection"]),
                                  "residual_corrected":dict(professional["window_detection"])})


def read_minute_csv(path):
    from app.adapters.model_features import CHANNEL_NAMES
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", *CHANNEL_NAMES]:
            raise ValueError("训练CSV列顺序与六通道合同不一致")
        for item in reader:
            stamp = datetime.fromisoformat(item["date"])
            if stamp.utcoffset() is None:
                stamp = stamp.replace(tzinfo=_ZONE)
            values = [float(item[name]) for name in CHANNEL_NAMES]
            if any(not isfinite(v) for v in values):
                raise ValueError("训练CSV含非有限值")
            if rows and stamp-datetime.fromisoformat(rows[-1]["event_time"]) != timedelta(seconds=60):
                raise ValueError("训练CSV分钟序列不连续或重复")
            rows.append(dict(event_time=stamp.isoformat(),values=values))
    if not rows:
        raise ValueError("训练CSV为空")
    return rows


def _write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state_digest(model):
    sha = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        sha.update(name.encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def train(january, february, legacy_source, output, report_dir, epochs=12, patience=3, batch_size=64, threads=4):
    import numpy as np
    import torch
    import lightgbm as lgb
    from app.adapters.gas_forecast import GasForecastAdapter
    from app.adapters.model_features import aggregate_five_minute, residual_vector, FORECAST_FEATURE_NAMES, RESIDUAL_FEATURE_NAMES, FEATURE_PIPELINE_VERSION

    started = time.monotonic()
    torch.set_num_threads(threads)
    january, february, output, report_dir = map(Path,(january,february,output,report_dir))
    if output.exists() and any(output.iterdir()):
        raise ValueError("模型训练输出目录须为空，已有版本保持不变")
    report_dir.mkdir(parents=True,exist_ok=True)
    rows_jan, rows_feb = read_minute_csv(january),read_minute_csv(february)
    for rows,month in [(rows_jan,1),(rows_feb,2)]:
        if any((datetime.fromisoformat(r["event_time"]).year,datetime.fromisoformat(r["event_time"]).month)!=(2022,month) for r in rows):
            raise ValueError("训练数据须为已确认31002的2022年1月/2月数据")
    buckets_jan,buckets_feb = aggregate_five_minute(rows_jan),aggregate_five_minute(rows_feb)
    samples_jan,samples_feb = build_windows(buckets_jan),build_windows(buckets_feb)
    point = lambda day: datetime.fromisoformat(day+"T00:00:00+08:00")
    boundaries = dict(fed_train=(point("2022-01-01"),point("2022-01-20")),
                      fed_validation=(point("2022-01-20"),point("2022-01-23")),
                      residual_train=(point("2022-01-23"),point("2022-01-27")),
                      residual_validation=(point("2022-01-27"),point("2022-02-01")))
    splits = {name:select_split(samples_jan,*bounds) for name,bounds in boundaries.items()}
    splits["test"] = samples_feb
    if any(not samples for samples in splits.values()):
        raise ValueError("时间划分有空分区")
    means,scales = fit_scaler(buckets_jan,point("2022-01-20"))
    architecture = dict(version="Fourier",mode_select="random",modes=16,seq_len=90,label_len=45,pred_len=12,
                        enc_in=6,dec_in=6,c_out=1,d_model=64,n_heads=8,e_layers=2,d_layers=1,d_ff=128,
                        moving_avg=[25],dropout=.1,embed="timeF",freq="h",activation="gelu",output_attention=False,
                        seed=2021,time_feature_offset_seconds=240,time_feature_timezone="+08:00")
    manifest = dict(kind="gas_forecast",model_id="fedformer-lgbm-31002",version="trained-20261103-v1",
                    source_id="multisensor-31002-2022",face_id="31002",aggregation_seconds=300,input_window=90,
                    horizon_seconds=3600,unit="%",feature_names=FORECAST_FEATURE_NAMES,
                    residual_feature_names=RESIDUAL_FEATURE_NAMES,residual_target="physical_error",residual_selection_mode="v1",
                    legacy_source_root=str(Path(legacy_source).resolve()),architecture=architecture,decision_offset=None,
                    feature_pipeline=FEATURE_PIPELINE_VERSION,
                    training_range=dict(start_at=rows_jan[0]["event_time"],end_at=point("2022-02-01").isoformat()),
                    training_data=[dict(sha256=_sha(january),source_id="multisensor-31002-2022",file_id=_sha(january),file_name=january.name)],
                    artifacts={})
    adapter = object.__new__(GasForecastAdapter)
    adapter.manifest = manifest
    model = adapter._build_model(torch,np)
    initial_digest = _state_digest(model)
    output.mkdir(parents=True,exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=.0005,weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=1)
    criterion = torch.nn.MSELoss()
    mean_array,scale_array = np.asarray(means,dtype=np.float32),np.asarray(scales,dtype=np.float32)
    marks_cache = {}
    def tensors(samples):
        x = (np.asarray([s["values"] for s in samples],dtype=np.float32)-mean_array)/scale_array
        y = (np.asarray([s["targets"] for s in samples],dtype=np.float32)-means[-1])/scales[-1]
        enc_marks,dec_marks = [],[]
        for sample in samples:
            history = [datetime.fromisoformat(t) for t in sample["times"]]
            future = [sample["origin"]+i*_STEP for i in range(12)]
            for t in history+future:
                if t not in marks_cache:
                    marks_cache[t] = time_marks([t])[0]
            enc_marks.append([marks_cache[t] for t in history])
            dec_marks.append([marks_cache[t] for t in history[-45:]+future])
        decoder = np.concatenate([x[:,-45:,-1:],np.zeros((len(x),12,1),dtype=np.float32)],axis=1)
        return torch.utils.data.TensorDataset(*[torch.from_numpy(a) for a in
                [x,np.asarray(enc_marks,dtype=np.float32),decoder,np.asarray(dec_marks,dtype=np.float32),y]])
    datasets = {key:tensors(values) for key,values in splits.items()}
    loader = lambda data,shuffle=False: torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=shuffle,
                   generator=torch.Generator().manual_seed(2021),num_workers=0)
    history,best_loss,best_epoch,stale,steps = [],float("inf"),0,0,0
    print(json.dumps(dict(stage="training_start",samples={k:len(v) for k,v in splits.items()},architecture=architecture)),flush=True)
    for epoch in range(1,epochs+1):
        epoch_start = time.monotonic()
        model.train()
        train_loss,train_count = 0.,0
        for x,x_mark,decoder,dec_mark,y in loader(datasets["fed_train"],True):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x,x_mark,decoder,dec_mark)[:,:,-1]
            loss = criterion(prediction,y)
            if not torch.isfinite(loss):
                raise ValueError("FEDformer训练损失出现非有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            train_loss += loss.item()*len(x);train_count += len(x);steps += 1
        model.eval()
        val_loss,val_count = 0.,0
        with torch.no_grad():
            for x,x_mark,decoder,dec_mark,y in loader(datasets["fed_validation"]):
                loss = criterion(model(x,x_mark,decoder,dec_mark)[:,:,-1],y)
                val_loss += loss.item()*len(x);val_count += len(x)
        val_loss /= val_count
        row = dict(epoch=epoch,train_standardized_mse=train_loss/train_count,validation_standardized_mse=val_loss,
                   seconds=time.monotonic()-epoch_start,optimizer_steps=steps)
        history.append(row)
        print(json.dumps(row),flush=True)
        with (report_dir/"training-history.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(row)+"\n")
        if val_loss < best_loss-1e-7:
            best_loss,best_epoch,stale = val_loss,epoch,0
            torch.save(model.state_dict(),output/"checkpoint.pth")
        else:
            stale += 1
        scheduler.step(val_loss)
        if stale >= patience:
            break
    model.load_state_dict(torch.load(output/"checkpoint.pth",weights_only=True,map_location="cpu"),strict=True)
    model.eval()
    final_digest = _state_digest(model)
    if steps == 0 or final_digest == initial_digest:
        raise ValueError("预测模型权重未经过训练更新")
    def base_predictions(key):
        batches = []
        with torch.no_grad():
            for x,x_mark,decoder,dec_mark,y in loader(datasets[key]):
                batches.append(model(x,x_mark,decoder,dec_mark)[:,:,-1].cpu().numpy()*scales[-1]+means[-1])
        return np.concatenate(batches)
    base = {key:base_predictions(key) for key in ["residual_train","residual_validation","test"]}
    truth = {key:np.asarray([s["targets"] for s in splits[key]],dtype=float) for key in base}
    residual_inputs = {key:np.asarray([residual_vector(s["values"]) for s in splits[key]],dtype=float) for key in base}
    boosters,residual_history = [],[]
    for col in range(12):
        residual = truth["residual_train"][:,col]-base["residual_train"][:,col]
        selected = residual>-.05
        if selected.sum()<10:
            raise ValueError("v1残差训练样本不足")
        val_residual = truth["residual_validation"][:,col]-base["residual_validation"][:,col]
        params = dict(objective="regression",metric="l2",verbosity=-1,num_leaves=15,learning_rate=.05,
                      min_data_in_leaf=20,lambda_l2=1.,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,
                      seed=2021+col,num_threads=threads,deterministic=True,force_col_wise=True)
        booster = lgb.train(params,lgb.Dataset(residual_inputs["residual_train"][selected],label=residual[selected]),
                            num_boost_round=100,valid_sets=[lgb.Dataset(residual_inputs["residual_validation"],label=val_residual)],
                            callbacks=[lgb.early_stopping(15,verbose=False)])
        path = output/f"residual_{col+1:02}.txt"
        path.write_text(booster.model_to_string(num_iteration=booster.best_iteration),encoding="utf-8")
        boosters.append(booster)
        residual_history.append(dict(lead_seconds=(col+1)*300,selected_samples=int(selected.sum()),
                                     candidate_samples=len(selected),best_iteration=booster.best_iteration))
    _write_json(output/"scaler.json",dict(feature_names=FORECAST_FEATURE_NAMES,mean=means,scale=scales,
                                       fitted_before=point("2022-01-20").isoformat()))
    for name,path in [("checkpoint",output/"checkpoint.pth"),("scaler",output/"scaler.json")]+[(f"residual_{i:02}",output/f"residual_{i:02}.txt") for i in range(1,13)]:
        manifest["artifacts"][name] = dict(path=path.name,sha256=_sha(path))
    manifest["training_protocol"] = dict(architecture_variant="compact_cpu_v1",feature_pipeline=FEATURE_PIPELINE_VERSION,
          epochs_completed=len(history),best_epoch=best_epoch,optimizer_steps=steps,initial_parameter_digest=initial_digest,
          trained_parameter_digest=final_digest,scaler_fit_end=point("2022-01-20").isoformat(),
          partitions={k:dict(start_at=a.isoformat(),end_at=b.isoformat()) for k,(a,b) in boundaries.items()},
          holdout="2022-02, excluded from every fit and model-selection decision")
    _write_json(output/"manifest.json",manifest)
    evaluation = {}
    for key in ("residual_validation","test"):
        corrected = base[key]+np.column_stack([booster.predict(residual_inputs[key],num_iteration=booster.best_iteration) for booster in boosters])
        persistence = np.asarray([[sample["values"][-1][-1]]*12 for sample in splits[key]])
        evaluation[key] = {name:forecast_metrics(truth[key].tolist(),pred.tolist()) for name,pred in
                           [("persistence",persistence),("base_prediction",base[key]),("residual_corrected",corrected)]}
        if key == "test":
            np.savez_compressed(report_dir/"holdout-predictions.npz",origins=np.asarray([s["origin"].isoformat() for s in splits[key]]),
                                truth=truth[key],persistence=persistence,base_prediction=base[key],residual_corrected=corrected)
    assessment = validation_assessment(evaluation)
    manifest.update(deployment_stage="experimental",validation_assessment=assessment)
    _write_json(output/"manifest.json",manifest)
    report = dict(status="completed",model_id=manifest["model_id"],model_version=manifest["version"],
                  trained_weights=True,total_seconds=time.monotonic()-started,training_history=history,
                  sample_counts={k:len(v) for k,v in splits.items()},residual_training=residual_history,
                  source_rows=dict(january=len(rows_jan),february=len(rows_feb)),source_lineage="processed",
                  training_file=dict(name=january.name,sha256=_sha(january)),test_file=dict(name=february.name,sha256=_sha(february)),
                  evaluation=evaluation,threshold_scope="0.65 is original algorithm evaluation threshold; source T0 maxima 0.75, no observed >=0.80 samples",
                  metrics_scope="overlapping forecast interval/window pairs; independent operational events are not deduplicated here",
                  deployment_stage="experimental",validation_assessment=assessment,
                  parameter_digests=dict(initial=initial_digest,trained=final_digest),output=str(output),
                  partitions=manifest["training_protocol"]["partitions"])
    _write_json(report_dir/"training-report.json",report)
    ready = GasForecastAdapter(output).inspect()
    if ready["status"] != "ready":
        raise ValueError("训练产物未通过封装加载检查")
    first = splits["test"][0]
    request = dict(as_of=first["origin"].isoformat(),unit="%",feature_names=FORECAST_FEATURE_NAMES,
                   times=first["times"],values=first["values"],residual_feature_names=RESIDUAL_FEATURE_NAMES,
                   residual_values=residual_vector(first["values"]))
    _write_json(report_dir/"first-holdout-request.json",request)
    packaged = GasForecastAdapter(output).predict(request)
    _write_json(report_dir/"first-holdout-response.json",packaged)
    _write_json(report_dir/"sample_request.json",request)
    _write_json(report_dir/"expected_outputs.json",packaged)
    print(json.dumps(dict(status="completed",output=str(output),best_epoch=best_epoch,total_seconds=report["total_seconds"],
                         test_metrics={k:v["mae"] for k,v in evaluation["test"].items()})),flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="31002多源FEDformer和物理残差训练")
    parser.add_argument("--january",type=Path,required=True)
    parser.add_argument("--february",type=Path,required=True)
    parser.add_argument("--legacy-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--patience",type=int,default=3)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args(argv)
    if min(args.epochs,args.patience,args.batch_size,args.threads)<=0:
        parser.error("训练参数须为正整数")
    train(args.january,args.february,args.legacy_source,args.output,args.report_dir,
          args.epochs,args.patience,args.batch_size,args.threads)


if __name__ == "__main__":
    main()
