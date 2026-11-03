"""FEDformer + LightGBM inference for trusted, versioned local bundles.

The heavy runtime and legacy imports belong in the isolated model worker.
Inputs are prepared training-order features, not raw sensor measurements.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib
import importlib.util
from math import isfinite
from numbers import Real
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.adapters.model_bundle import ModelBundle, ModelNotReady, read_json

__all__ = ["GasForecastAdapter"]

_ARTIFACTS = ["checkpoint", "scaler"] + [f"residual_{i:02}" for i in range(1, 13)]
_DEPENDENCIES = ("torch", "numpy", "lightgbm", "scipy", "einops", "sympy", "matplotlib")
_REQUEST_FIELDS = {
    "as_of", "unit", "feature_names", "times", "values",
    "residual_feature_names", "residual_values", "horizon_seconds",
}


def _datetime(value):
    if not isinstance(value, str):
        raise ValueError("模型输入时间须为带时区的ISO字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("模型输入时间格式无效") from error
    if parsed.utcoffset() is None:
        raise ValueError("模型输入时间缺少时区")
    return parsed.astimezone(timezone.utc)


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


def _names(value, name):
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v.strip() for v in value) or len(set(value)) != len(value):
        raise ValueError(f"{name}须为不重复的训练特征名称列表")


def _time_zone(name):
    if name == "UTC":
        return timezone.utc
    if isinstance(name, str) and re.fullmatch(r"[+-]\d{2}:\d{2}", name):
        hours, minutes = map(int, name[1:].split(":"))
        if hours <= 23 and minutes <= 59:
            return timezone(timedelta(minutes=(hours * 60 + minutes) * (-1 if name[0] == "-" else 1)))
    try:
        return ZoneInfo(name)
    except (TypeError, ValueError, ZoneInfoNotFoundError) as error:
        raise ValueError("训练时间特征时区无效") from error


class GasForecastAdapter:
    def __init__(self, bundle_dir):
        self.bundle = ModelBundle(bundle_dir, expected_kind="gas_forecast")
        self.manifest = self.bundle.manifest

    def inspect(self):
        result = {
            "status": "not_ready", "model_id": self.manifest.get("model_id"),
            "model_version": self.manifest.get("version"), "missing_fields": [],
        }
        try:
            artifacts = self.bundle.require_artifacts(_ARTIFACTS)
            self._validate_manifest()
            self._scaler(artifacts["scaler"])
            self._source_root()
        except ModelNotReady as error:
            result["missing_fields"] = error.missing_fields
        except ValueError as error:
            result["missing_fields"] = [f"manifest: {error}"]
        else:
            try:
                self._load_predictors(artifacts)
            except ModelNotReady as error:
                result["missing_fields"] = error.missing_fields
            except Exception as error:
                result.update({
                    "missing_fields": ["model_artifacts"], "error_code": "invalid_model_artifacts",
                    "error_type": type(error).__name__,
                })
            else:
                result["status"] = "ready"
        return result

    def _validate_manifest(self):
        manifest = self.manifest
        if manifest.get("aggregation_seconds") != 300 or manifest.get("input_window") != 90 or manifest.get("horizon_seconds") != 3600:
            raise ValueError("FEDformer封装要求90个5分钟输入桶和12个5分钟预测区间")
        if manifest.get("unit", "%") != "%":
            raise ValueError("预测模型浓度单位须为%")
        _names(manifest.get("feature_names"), "feature_names")
        _names(manifest.get("residual_feature_names"), "residual_feature_names")
        if manifest.get("residual_target") != "physical_error" or manifest.get("residual_selection_mode") != "v1":
            raise ValueError("残差目标须为原v1的真实浓度减基础预测，禁止混入阈值偏移")
        architecture = manifest.get("architecture")
        required = {
            "version", "mode_select", "modes", "seq_len", "label_len", "pred_len",
            "enc_in", "dec_in", "c_out", "d_model", "n_heads", "e_layers", "d_layers",
            "d_ff", "moving_avg", "dropout", "embed", "freq", "activation",
            "output_attention", "seed", "time_feature_offset_seconds", "time_feature_timezone",
        }
        if not isinstance(architecture, dict) or required - architecture.keys():
            raise ValueError("缺少完整架构或训练时间特征配置")
        if architecture["version"] != "Fourier" or architecture["mode_select"] not in ("random", "low"):
            raise ValueError("当前封装使用Fourier FEDformer及明确的频率选择策略")
        for name in ("modes", "seq_len", "label_len", "pred_len", "enc_in", "dec_in", "c_out", "d_model", "n_heads", "e_layers", "d_layers", "d_ff"):
            if type(architecture[name]) is not int or architecture[name] <= 0:
                raise ValueError(f"架构字段{name}须为正整数")
        if architecture["seq_len"] != 90 or architecture["pred_len"] != 12 or architecture["label_len"] != 45:
            raise ValueError("模型架构窗口与清单不一致")
        size = len(manifest["feature_names"])
        if architecture["enc_in"] != size or architecture["dec_in"] != size or architecture["c_out"] != 1 or architecture["n_heads"] != 8:
            raise ValueError("模型架构维度与训练特征或原Fourier实现不一致")
        if architecture["d_model"] % 8 or architecture["embed"] != "timeF" or architecture["freq"] not in ("h", "t", "s"):
            raise ValueError("模型通道或时间编码配置无效")
        if type(architecture["seed"]) is not int or not 0 <= architecture["seed"] < 2**32:
            raise ValueError("模型须记录有效的训练随机种子")
        offset = architecture["time_feature_offset_seconds"]
        if type(offset) is not int or not 0 <= offset < 300:
            raise ValueError("训练时间特征偏移须位于输入桶内")
        _time_zone(architecture["time_feature_timezone"])
        kernel = architecture["moving_avg"]
        kernels = kernel if isinstance(kernel, list) else [kernel]
        if not kernels or any(type(v) is not int or v <= 0 for v in kernels):
            raise ValueError("移动平均窗口无效")
        if not _finite(architecture["dropout"]) or not 0 <= architecture["dropout"] < 1 or type(architecture["output_attention"]) is not bool:
            raise ValueError("模型架构参数无效")
        training_range = manifest.get("training_range")
        if not isinstance(training_range, dict) or "start_at" not in training_range or "end_at" not in training_range:
            raise ValueError("模型须记录训练时间范围")
        if _datetime(training_range["start_at"]) >= _datetime(training_range["end_at"]):
            raise ValueError("训练时间范围无效")
        training = manifest.get("training_data")
        if not isinstance(training, list) or not training or any(
            not isinstance(item, dict) or not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", item["sha256"])
            for item in training
        ):
            raise ValueError("模型须记录训练源文件SHA-256")
        decision = manifest.get("decision_offset")
        if decision is not None and (not isinstance(decision, dict) or any(not _finite(decision.get(k)) for k in ("true_threshold", "best_model_threshold"))):
            raise ValueError("决策阈值偏移配置无效")

    def _source_root(self):
        value = self.manifest.get("legacy_source_root")
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ModelNotReady(["legacy_source_root"])
        root = Path(value).resolve()
        if not (root / "models" / "FEDformer.py").is_file():
            raise ModelNotReady(["legacy_source_root.models.FEDformer"])
        return root

    def _validate_request(self, request):
        if not isinstance(request, dict) or request.keys() - _REQUEST_FIELDS:
            raise ValueError("模型请求含未知字段")
        manifest = self.manifest
        if request.get("unit") != "%":
            raise ValueError("模型请求浓度单位须为%")
        if request.get("feature_names") != manifest["feature_names"] or request.get("residual_feature_names") != manifest["residual_feature_names"]:
            raise ValueError("模型请求特征顺序与训练清单不一致")
        cutoff = _datetime(request.get("as_of"))
        if cutoff.timestamp() % 300:
            raise ValueError("预测时钟须位于闭合5分钟桶边界")
        if _datetime(manifest["training_range"]["end_at"]) > cutoff:
            raise ValueError("模型训练结束时间晚于预测时钟")
        times, values = request.get("times"), request.get("values")
        if not isinstance(times, list) or not isinstance(values, list) or len(times) != 90 or len(values) != 90:
            raise ValueError("模型需要90个闭合输入桶")
        parsed = [_datetime(value) for value in times]
        start = cutoff - timedelta(seconds=90 * 300)
        for index, (timestamp, row) in enumerate(zip(parsed, values)):
            if timestamp != start + timedelta(seconds=300 * index):
                raise ValueError("模型输入桶不连续或超过预测时钟")
            if not isinstance(row, list) or len(row) != len(manifest["feature_names"]) or any(not _finite(v) for v in row):
                raise ValueError("模型输入含无效特征值或维度")
        residual = request.get("residual_values")
        if not isinstance(residual, list) or len(residual) != len(manifest["residual_feature_names"]) or any(not _finite(v) for v in residual):
            raise ValueError("残差输入含无效特征值或维度")
        horizon = request.get("horizon_seconds", 3600)
        if type(horizon) is not int or not 0 < horizon <= 3600 or horizon % 300:
            raise ValueError("预测时长须为模型范围内的5分钟整数倍")
        return cutoff, parsed, horizon

    def _scaler(self, path):
        scaler = read_json(path)
        if not isinstance(scaler, dict) or scaler.get("feature_names") != self.manifest["feature_names"]:
            raise ValueError("标准化参数的训练特征顺序不一致")
        size = len(self.manifest["feature_names"])
        for key in ("mean", "scale"):
            if not isinstance(scaler.get(key), list) or len(scaler[key]) != size or any(not _finite(v) for v in scaler[key]):
                raise ValueError("标准化参数含无效数值或维度")
        if any(v <= 0 for v in scaler["scale"]):
            raise ValueError("标准化尺度须为正数")
        return scaler

    def predict(self, request):
        artifacts = self.bundle.require_artifacts(_ARTIFACTS)
        self._validate_manifest()
        cutoff, times, horizon = self._validate_request(request)
        scaler = self._scaler(artifacts["scaler"])
        values = [[(value - mean) / scale for value, mean, scale in zip(row, scaler["mean"], scaler["scale"])] for row in request["values"]]
        base_model, residual_models = self._load_predictors(artifacts)
        standardized = base_model(values, times, cutoff)
        if not isinstance(standardized, list) or len(standardized) != 12 or any(not _finite(v) for v in standardized):
            raise ValueError("FEDformer输出含无效数值或区间数量")
        base = [float(value * scaler["scale"][-1] + scaler["mean"][-1]) for value in standardized]
        if len(residual_models) != 12:
            raise ValueError("残差模型数量须为12")
        residual = []
        for model in residual_models:
            output = model.predict([request["residual_values"]])
            try:
                output = list(output)
            except TypeError as error:
                raise ValueError("LightGBM残差输出无效") from error
            if len(output) != 1 or not _finite(output[0]):
                raise ValueError("LightGBM残差输出无效")
            residual.append(float(output[0]))
        corrected = [raw + delta for raw, delta in zip(base, residual)]
        series = [("base_prediction", base), ("residual", residual), ("residual_corrected", corrected)]
        decision = self.manifest.get("decision_offset")
        if decision is not None:
            offset = decision["true_threshold"] - decision["best_model_threshold"]
            series.append(("decision_score", [value + offset for value in corrected]))
        outputs = []
        for stage, numbers in series:
            if any(not _finite(v) for v in numbers):
                raise ValueError("模型输出含非有限数值")
            outputs.append({
                "prediction_stage": stage,
                "quantity_semantics": "decision_score" if stage == "decision_score" else "physical_forecast",
                "unit": "dimensionless" if stage == "decision_score" else "%",
                "target_statistic": "interval_maximum",
                "interval_values": [
                    {"start_at": (cutoff + timedelta(seconds=index * 300)).isoformat(),
                     "end_at": (cutoff + timedelta(seconds=(index + 1) * 300)).isoformat(), "value": value}
                    for index, value in enumerate(numbers[:horizon // 300])
                ],
            })
        return {
            "status": "completed", "model_id": self.manifest["model_id"], "model_version": self.manifest["version"],
            "as_of": cutoff.isoformat(), "target_statistic": "interval_maximum", "outputs": outputs,
            "training_range": deepcopy(self.manifest["training_range"]),
            "training_data": deepcopy(self.manifest["training_data"]),
            "residual_target": self.manifest["residual_target"],
            "residual_selection_mode": self.manifest["residual_selection_mode"],
            "decision_offset": deepcopy(decision),
        }

    def _build_model(self, torch, numpy):
        root = self._source_root()
        for name in ("models", "layers", "utils"):
            module = sys.modules.get(name)
            if module is not None:
                location = getattr(module, "__file__", None)
                paths = [Path(p).resolve() for p in getattr(module, "__path__", [])]
                if (location and not Path(location).resolve().is_relative_to(root)) or any(not p.is_relative_to(root) for p in paths):
                    raise ValueError("模型运行器已加载其他来源的同名模块")
        sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("models.FEDformer")
        finally:
            sys.path.pop(0)
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError("FEDformer代码来源与清单不一致")
        config = self.manifest["architecture"]
        random.seed(config["seed"])
        numpy.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        module.device = torch.device("cpu")
        return module.Model(SimpleNamespace(**config)).to("cpu")

    def _load_predictors(self, artifacts):
        missing = [f"dependencies.{name}" for name in _DEPENDENCIES if importlib.util.find_spec(name) is None]
        if missing:
            raise ModelNotReady(missing)
        import numpy
        import torch
        import lightgbm

        model = self._build_model(torch, numpy)
        state = torch.load(artifacts["checkpoint"], map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        # Python handles Windows Unicode paths; the native C file API does not.
        boosters = [lightgbm.Booster(model_str=artifacts[f"residual_{i:02}"].read_text(encoding="utf-8-sig"))
                    for i in range(1, 13)]
        if any(booster.num_feature() != len(self.manifest["residual_feature_names"]) for booster in boosters):
            raise ValueError("残差模型维度与训练特征清单不一致")
        config = self.manifest["architecture"]
        zone = _time_zone(config["time_feature_timezone"])
        offset = timedelta(seconds=config["time_feature_offset_seconds"])

        def marks(times):
            encoded = []
            for timestamp in times:
                local = (timestamp + offset).astimezone(zone)
                fields = [local.hour / 23 - 0.5, local.weekday() / 6 - 0.5, (local.day - 1) / 30 - 0.5, (local.timetuple().tm_yday - 1) / 365 - 0.5]
                if config["freq"] in ("t", "s"):
                    fields.insert(0, local.minute / 59 - 0.5)
                if config["freq"] == "s":
                    fields.insert(0, local.second / 59 - 0.5)
                encoded.append(fields)
            return encoded

        def predict_base(values, times, cutoff):
            history = config["label_len"]
            future = [cutoff + timedelta(seconds=300 * index) for index in range(12)]
            decoder = [[row[-1]] for row in values[-history:]] + [[0.0] for _ in range(12)]
            tensor = lambda rows: torch.tensor([rows], dtype=torch.float32, device="cpu")
            with torch.inference_mode():
                output = model(tensor(values), tensor(marks(times)), tensor(decoder), tensor(marks(times[-history:] + future)))
            if config["output_attention"]:
                output = output[0]
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 12:
                raise ValueError("FEDformer输出维度无效")
            return output[0, :, -1].detach().cpu().tolist()

        return predict_base, boosters
