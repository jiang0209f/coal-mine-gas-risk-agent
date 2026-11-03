"""Build trained-model features from authenticated, source-bound observations."""

from copy import deepcopy
from datetime import timedelta, timezone, datetime
from math import floor
from uuid import uuid4

from app.adapters.model_bundle import ModelNotReady
from app.adapters.model_features import (
    CHANNEL_NAMES, FEATURE_PIPELINE_VERSION, FORECAST_FEATURE_NAMES, RESIDUAL_FEATURE_NAMES,
    forecast_request, attribution_request, timestamp,
)
from app.schemas.common import DomainError, utcstr, now_utc


class TrainedModelsPipeline:
    def __init__(self, store, data, runtime):
        self.store, self.data, self.runtime = store, data, runtime

    def _package(self, context, kind, model_id=None, version=None):
        context.actor.require_role("ventilation", "gas_governance", "safety_management")
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("数据源不存在", "not_found", 404)
        context.actor.require_face(source["face_id"])
        packages = [p for p in self.runtime.describe_packages() if p["kind"] == kind
                    and p["source_id"] == context.source_id and p["face_id"] == source["face_id"]
                    and (model_id is None or p["model_id"] == model_id)
                    and (version is None or p["model_version"] == version)]
        if not packages:
            raise ModelNotReady(["trained_model_for_source"])
        if len(packages) != 1:
            raise ModelNotReady(["unambiguous_trained_model_version"])
        package = packages[0]
        spec = self.runtime.get_spec(package["model_id"], package["model_version"])
        manifest = spec["manifest"]
        points = manifest.get("feature_points")
        if not isinstance(points, dict) or set(points) != set(CHANNEL_NAMES) or len(set(points.values())) != 6:
            raise ModelNotReady(["six_channel_point_bindings"])
        for point_id in points.values():
            point = self.store.get("point", point_id)
            if point is None or point.get("source_id") != context.source_id or point.get("face_id") != source["face_id"]:
                raise DomainError("模型特征测点与会话来源不一致", "forbidden", 403)
            if point.get("mapping_status") != "confirmed" or point.get("lineage") != "processed":
                raise ModelNotReady(["confirmed_processed_feature_points"])
        return package, manifest, points

    def _minutes(self, context, points, start, end, visible_at):
        frozen = context.model_copy(update={"as_of": visible_at})
        columns, references = [], []
        for name in CHANNEL_NAMES:
            point_id = points[name]
            result = self.data.query_series(frozen, point_id, start, end, aggregation_seconds=None,
                                            fill_max_seconds=0, limit=1000, offset=0)
            values = {}
            for row in result["rows"]:
                moment = timestamp(row["event_time"])
                flags = row.get("quality_flags", [])
                if moment in values or row.get("value") is None or any(f in flags for f in ("invalid_value", "duplicate_timestamp", "forward_filled")):
                    raise ModelNotReady(["valid_unique_minute_features"])
                values[moment] = row["value"]
                references.append({key: row[key] for key in ("id", "point_id", "source_file_id", "row_number", "event_time", "available_at") if key in row})
            columns.append(values)
        common = set(columns[0])
        if any(set(column) != common for column in columns[1:]):
            raise ModelNotReady(["aligned_six_channel_minute_history"])
        rows = [dict(event_time=t.isoformat(), values=[column[t] for column in columns]) for t in sorted(common)]
        return rows, references

    def forecast(self, context, point_id, model, horizon_seconds):
        package, manifest, points = self._package(context, "gas_forecast", model["model_id"], model["version"])
        if (point_id != points[CHANNEL_NAMES[0]] or model.get("feature_points") != points
                or model.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION
                or manifest.get("feature_pipeline_version") != FEATURE_PIPELINE_VERSION
                or manifest.get("feature_names") != FORECAST_FEATURE_NAMES
                or manifest.get("residual_feature_names") != RESIDUAL_FEATURE_NAMES):
            raise DomainError("模型登记与训练特征清单不一致")
        cutoff = datetime.fromtimestamp(floor(context.as_of.timestamp()/300)*300, timezone.utc)
        if timestamp(manifest["training_range"]["end_at"]) > cutoff:
            raise ModelNotReady(["training_cutoff_not_after_as_of"])
        rows, references = self._minutes(context, points, cutoff-timedelta(minutes=450), cutoff, cutoff)
        request = forecast_request(rows, cutoff)
        request["horizon_seconds"] = horizon_seconds
        result = self.runtime.infer(package["model_id"], package["model_version"], request)
        if result["status"] == "not_ready":
            raise ModelNotReady(result.get("missing_fields") or ["trained_prediction_artifacts"])
        return {**result, "input_refs": references, "input_origin": "server_observations",
                "feature_pipeline_version": FEATURE_PIPELINE_VERSION}

    def attribute_event(self, context, event, fragments):
        def unavailable(fields):
            return dict(status="not_ready", missing_fields=fields, causal_status="evidence_only")
        if event.get("event_type") != "observed_exceedance":
            return unavailable(["observed_event_required"])
        try:
            package, manifest, points = self._package(context, "gas_attribution")
            if event["point_id"] != points[CHANNEL_NAMES[0]]:
                return unavailable(["target_point_binding"])
            candidates = [r for r in fragments if r.get("point_id") == event["point_id"]
                          and r.get("value") is not None and r["value"] >= .65
                          and timestamp(r["event_time"]) <= context.as_of]
            if not candidates:
                return unavailable(["observed_target_at_model_threshold"])
            target = min(candidates, key=lambda r: timestamp(r["event_time"]))
            origin = timestamp(target["event_time"]) - timedelta(seconds=manifest["horizon_seconds"])
            # Production manifests keep automatic attribution disabled until its audit is accepted.
            if manifest.get("automatic_attribution", False) is not True:
                return unavailable(["automatic_attribution_validation"])
            rows, references = self._minutes(context, points, origin-timedelta(minutes=60), origin+timedelta(microseconds=1), origin)
            result = self.runtime.infer(package["model_id"], package["model_version"], attribution_request(rows, origin))
            if result["status"] != "completed":
                return result
            run = {**result, "run_id": str(uuid4()), "session_id": context.session_id, "source_id": context.source_id,
                   "face_id": package["face_id"], "event_id": event["event_id"], "event_revision": event["event_revision"],
                   "target_observation_id": target["id"], "target_time": target["event_time"],
                   "target_policy": "first_model_threshold_observation_in_event", "input_as_of": utcstr(origin),
                   "as_of": utcstr(context.as_of), "input_origin": "server_observations", "input_refs": references,
                   "created_at": utcstr(now_utc())}
            self.store.put("attribution_run", run["run_id"], run)
            return deepcopy(run)
        except ModelNotReady as error:
            return unavailable(error.missing_fields)
