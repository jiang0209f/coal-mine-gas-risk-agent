"""Register trained 31002 packages and their source-bound historical signals."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from sqlalchemy import func, select

from app.adapters.model_bundle import ModelBundle, read_json
from app.adapters.model_features import CHANNEL_NAMES, FEATURE_PIPELINE_VERSION
from app.db import Store, observations
from app.schemas.common import Actor, DomainError
from app.scripts.train_forecast import read_minute_csv
from app.services.data.service import DataService
from app.services.prediction.service import PredictionService

SOURCE_ID = "multisensor-31002-2022"
FACE_ID = "31002"
FEATURE_POINTS = dict(zip(CHANNEL_NAMES, ["31002:multi:T0", "31002:multi:wind_speed", "31002:multi:shearer_speed",
                                         "31002:multi:shearer_methane", "31002:multi:left_current", "31002:multi:right_current"]))
_FORECAST_ARTIFACTS = ["checkpoint", "scaler", *[f"residual_{i:02}" for i in range(1,13)]]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid4().hex+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2),encoding="utf-8")
    temporary.replace(path)


def _preserve_file_once(path, contents):
    try:
        with Path(path).open("xb") as stream:
            stream.write(contents)
    except FileExistsError:
        pass


def _backup_once(database, report_dir):
    destination = report_dir/"before-trained-registration.sqlite"
    created = False
    if database.is_file() and not destination.exists():
        try:
            with destination.open("xb"):
                pass
        except FileExistsError:
            pass
        else:
            with sqlite3.connect(database.as_uri()+"?mode=ro",uri=True) as source:
                with sqlite3.connect(destination) as target:
                    source.backup(target)
            created = True
    if destination.exists():
        try:
            if destination.stat().st_size < 100:
                raise ValueError("已有数据库备份为空或损坏")
            with sqlite3.connect(destination.as_uri()+"?mode=ro",uri=True) as existing:
                valid = existing.execute("PRAGMA quick_check").fetchone()
            if valid != ("ok",):
                raise ValueError("已有数据库备份完整性检查失败")
        except sqlite3.Error as error:
            raise ValueError("已有数据库备份损坏；保留备份并停止接入") from error
    return dict(path=str(destination),created=created,sha256=_sha(destination) if destination.is_file() else None,
                source_existed=database.is_file())


def _point_specs():
    signals = ["methane", "wind_speed", "shearer_speed", "shearer_methane", "left_current", "right_current"]
    return [dict(point_id=FEATURE_POINTS[name],source_id=SOURCE_ID,face_id=FACE_ID,source_tag=name,name=name,
                 quantity=quantity,unit="%" if index in (0,3) else "source_native",unit_status="inferred",
                 mapping_status="confirmed",lineage="processed")
            for index,(name,quantity) in enumerate(zip(CHANNEL_NAMES,signals))]


def _augment(manifest):
    result = deepcopy(manifest)
    bindings = dict(source_id=SOURCE_ID,face_id=FACE_ID,feature_points=FEATURE_POINTS,
                    feature_pipeline_version=FEATURE_PIPELINE_VERSION)
    for field,value in bindings.items():
        if field in result and result[field] != value:
            raise ValueError("已训练模型的来源或特征绑定冲突")
        result[field] = deepcopy(value)
    result["deployment_stage"] = "experimental"
    if result["kind"] == "gas_attribution":
        result["automatic_attribution"] = False
    return result


def _model_specs(forecast):
    shared = dict(face_id=FACE_ID,source_id=SOURCE_ID,point_ids=[FEATURE_POINTS[CHANNEL_NAMES[0]]],
                  target_quantity="methane",unit="%",aggregation_seconds=300,horizon_seconds=3600,
                  allowed_lineage=["processed"],target_statistic="interval_maximum")
    return [dict(shared,model_id=forecast["model_id"],version=forecast["version"],adapter_kind="fedformer_lgbm",
                 input_window=90,default_eligible=False,feature_pipeline_version=FEATURE_PIPELINE_VERSION,
                 feature_points=deepcopy(FEATURE_POINTS),deployment_stage="experimental"),
            dict(shared,model_id="persistence-5min-max-31002",version="validated-20261103-v1",adapter_kind="persistence",
                 input_window=1,input_statistic="max",default_eligible=True,
                 selection_basis="January validation concentration MAE; trained FEDformer remains experimental")]


def _matches_model(existing, desired):
    expected = dict(desired,model_type="baseline" if desired["adapter_kind"]=="persistence" else "professional")
    if desired["adapter_kind"] == "persistence":
        expected["purpose"] = "offline_pipeline_verification"
    return all(existing.get(key)==value for key,value in expected.items())


def _catalog_plan(catalog_path, bundles):
    catalog = read_json(catalog_path)
    if not isinstance(catalog,dict) or not isinstance(catalog.get("models"),list):
        raise ValueError("模型目录格式无效")
    result = deepcopy(catalog)
    identities = {}
    for entry in result["models"]:
        if not isinstance(entry,dict) or not isinstance(entry.get("bundle_dir"),str):
            raise ValueError("模型目录条目无效")
        path = (catalog_path.parent/entry["bundle_dir"]).resolve()
        manifest = read_json(path/"manifest.json")
        if not isinstance(manifest,dict) or not all(isinstance(manifest.get(k),str) and manifest[k] for k in ("model_id","version","kind")):
            raise ValueError("既有模型目录身份无效")
        key = (manifest["model_id"],manifest["version"])
        if key in identities:
            raise ValueError("既有模型目录存在重复编号与版本")
        identities[key] = (path,entry)
    appended = []
    for path,manifest in bundles:
        key = (manifest["model_id"],manifest["version"])
        if key in identities:
            existing_path,entry = identities[key]
            if existing_path != path or entry.get("source_id")!=SOURCE_ID or entry.get("face_id")!=FACE_ID:
                raise ValueError("同一模型版本已绑定其他目录或来源")
            continue
        entry = dict(bundle_dir=str(path),source_id=SOURCE_ID,face_id=FACE_ID)
        result["models"].append(entry)
        identities[key] = (path,entry)
        appended.append(dict(model_id=key[0],version=key[1]))
    return result,appended


def _source_count(store, source_id):
    with store.transaction() as connection:
        return connection.execute(select(func.count()).select_from(observations).where(observations.c.source_id==source_id)).scalar_one()


def register_trained_models(january, february, forecast_bundle, attribution_bundle, runtime_dir, catalog, report_dir):
    january,february,forecast_bundle,attribution_bundle,runtime_dir,catalog,report_dir = [Path(p).resolve() for p in
        (january,february,forecast_bundle,attribution_bundle,runtime_dir,catalog,report_dir)]
    source_files = []
    for file,month in [(january,1),(february,2)]:
        rows = read_minute_csv(file)
        if any((datetime.fromisoformat(row["event_time"]).year,datetime.fromisoformat(row["event_time"]).month)!=(2022,month) for row in rows):
            raise ValueError("数据须为已确认31002的2022年1月和2月")
        source_files.append(dict(path=file,sha256=_sha(file),minute_rows=len(rows)))
    forecast_reader = ModelBundle(forecast_bundle,"gas_forecast")
    attribution_reader = ModelBundle(attribution_bundle,"gas_attribution")
    forecast_reader.require_artifacts(_FORECAST_ARTIFACTS)
    attribution_reader.require_artifacts(["classifier","training_metadata"])
    forecast,attribution = _augment(forecast_reader.manifest),_augment(attribution_reader.manifest)
    if not any(isinstance(item,dict) and item.get("sha256")==source_files[0]["sha256"] for item in forecast.get("training_data",[])):
        raise ValueError("January文件摘要与预测训练依据不符")
    assessment = forecast.get("validation_assessment",{})
    if assessment.get("default_candidate")!="persistence" or assessment.get("professional_mae_improved") is not False:
        raise ValueError("当前接入方案要求已有January验证支持持久性作为默认候选")
    catalog_after,appended = _catalog_plan(catalog,[(forecast_bundle,forecast),(attribution_bundle,attribution)])
    specs = _model_specs(forecast)
    report_dir.mkdir(parents=True,exist_ok=True)
    runtime_dir.mkdir(parents=True,exist_ok=True)
    database = runtime_dir/"app.db"
    backup = _backup_once(database,report_dir)
    store = Store(f"sqlite+pysqlite:///{database.as_posix()}")
    actor = Actor(user_id="local-trained-model-registration",roles=("safety_management",),face_ids=(FACE_ID,))
    try:
        source = store.get("source",SOURCE_ID)
        if source and (source.get("face_id")!=FACE_ID or source.get("timezone")!="+08:00"):
            raise ValueError("既有多源数据源的工作面或时区冲突")
        for point in _point_specs():
            existing = store.get("point",point["point_id"])
            if existing is not None and existing!=point:
                raise ValueError("既有多源测点元数据冲突")
        for model in specs:
            existing = store.get("model",f"{model['model_id']}:{model['version']}")
            if existing is not None and not _matches_model(existing,model):
                raise ValueError("已登记模型版本内容冲突")
        original_source,original_point = store.get("source","history-31002"),store.get("point","31002:T0")
        original_count = _source_count(store,"history-31002")
        data = DataService(store,runtime_dir/"imports")
        if source is None:
            source = data.register_source(SOURCE_ID,"31002多源历史数据（2022年1—2月）",FACE_ID,"+08:00")
        for point in _point_specs():
            data.register_point(**{k:v for k,v in point.items() if k!="face_id"})
        mapping = dict(time_column="date",value_columns=deepcopy(FEATURE_POINTS),timezone="+08:00",lineage="processed",header_row=1)
        imports=[]
        for file in source_files:
            registered = data.register_file(file["path"],actor)
            imported = data.import_history(actor,SOURCE_ID,registered["file_id"],mapping)
            imports.append(dict(imported,file_name=file["path"].name,sha256=file["sha256"],minute_rows=file["minute_rows"]))
        predictions = PredictionService(store,data)
        registered_models=[]
        for spec in specs:
            key=f"{spec['model_id']}:{spec['version']}"
            existing=store.get("model",key)
            registered_models.append(dict(model_id=spec["model_id"],version=spec["version"],created=existing is None,
                                          default_eligible=spec["default_eligible"]))
            if existing is None:
                predictions.register_model(actor,spec)
        original_after = _source_count(store,"history-31002")
        if original_count!=original_after or original_source!=store.get("source","history-31002") or original_point!=store.get("point","31002:T0"):
            raise ValueError("原始数据命名空间发生变化")
        count_after = _source_count(store,SOURCE_ID)
    finally:
        store.close()
    _preserve_file_once(report_dir/"catalog-before-registration.json",catalog.read_bytes())
    for name,path in [("forecast",forecast_bundle),("attribution",attribution_bundle)]:
        _preserve_file_once(report_dir/(name+"-manifest-before-registration.json"),(path/"manifest.json").read_bytes())
    _json(forecast_bundle/"manifest.json",forecast)
    _json(attribution_bundle/"manifest.json",attribution)
    _json(catalog,catalog_after)
    # Metadata bindings change, while every serialized model artifact remains intact.
    forecast_reader.require_artifacts(_FORECAST_ARTIFACTS)
    attribution_reader.require_artifacts(["classifier","training_metadata"])
    result = dict(status="completed",created_at=datetime.now(timezone.utc).isoformat(),source_id=SOURCE_ID,face_id=FACE_ID,
                  database_backup=backup,imports=imports,imported_rows_this_run=sum(item["imported_rows"] for item in imports),
                  source_observations_after=count_after,feature_points=deepcopy(FEATURE_POINTS),
                  feature_pipeline_version=FEATURE_PIPELINE_VERSION,models=registered_models,catalog_appended=appended,
                  original_namespace_unchanged=True,original_source_observations_before=original_count,
                  original_source_observations_after=original_after,automatic_attribution=False,
                  artifacts_unchanged=True,artifact_hashes={"forecast":deepcopy(forecast["artifacts"]),"attribution":deepcopy(attribution["artifacts"])},
                  catalog_sha256=_sha(catalog),manifest_sha256={"forecast":_sha(forecast_bundle/"manifest.json"),"attribution":_sha(attribution_bundle/"manifest.json")})
    _json(report_dir/"registration-report.json",result)
    with (report_dir/"registration-history.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description="备份并接入已训练31002模型和六通道历史数据")
    for field in ("january","february","forecast-bundle","attribution-bundle","runtime-dir","catalog","report-dir"):
        parser.add_argument("--"+field,type=Path,required=True)
    arguments=parser.parse_args(argv)
    try:
        result=register_trained_models(**vars(arguments))
    except (OSError,ValueError,DomainError) as error:
        print(json.dumps(dict(status="failed",error_type=type(error).__name__,message=str(error)),ensure_ascii=False))
        return 1
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
