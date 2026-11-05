"""Evaluate frozen synthetic support labels; imports no model until CLI execution."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import time
import warnings
from unittest.mock import patch

__all__ = ["load_benchmark", "evaluate_benchmark"]


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _verified_json(path, expected_sha):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError("评测数据SHA与冻结清单不一致")
    return json.loads(content)


def _validate(candidates, queries):
    if not isinstance(candidates, list) or not isinstance(queries, list):
        raise ValueError("评测候选与问题必须为列表")
    references = [row.get("reference_id") for row in candidates]
    if any(not isinstance(item, str) or not item for item in references) or len(set(references)) != len(references):
        raise ValueError("评测候选引用编号无效或重复")
    references = set(references)
    seen = set()
    for query in queries:
        query_id, label = query.get("query_id"), query.get("label")
        gold = query.get("supporting_reference_ids")
        background = query.get("background_relevant_reference_ids", [])
        hard_negative = query.get("hard_negative_reference_ids", [])
        expectation = query.get("retrieval_expectation", "supporting_passage" if label == "supported" else "no_supporting_passage")
        if (not isinstance(query_id, str) or not query_id or query_id in seen
                or not isinstance(query.get("query"), str) or not query["query"].strip()
                or label not in ("supported", "unsupported", "irrelevant")
                or not isinstance(gold, list) or not isinstance(background, list) or not isinstance(hard_negative, list)
                or not set(gold + background + hard_negative).issubset(references)
                or expectation not in ("supporting_passage", "no_supporting_passage", "background_allowed")
                or (label == "supported") != bool(gold)
                or (query.get("retrieval_expectation") == "background_allowed" and not background)):
            raise ValueError("评测金标、问题编号或支持引用无效")
        seen.add(query_id)


def load_benchmark(dataset_path, split="dev") -> dict:
    """Load approved frozen fixtures; dev does not open heldout question data."""
    if split not in ("dev", "heldout", "all"):
        raise ValueError("评测split必须为dev、heldout或all")
    path = Path(dataset_path).resolve()
    manifest_path = path.with_name("knowledge_retrieval_benchmark_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    main = _verified_json(path, manifest["files"]["dev_and_candidates"]["sha256"])
    queries = list(main["queries"]) if split in ("dev", "all") else []
    if split in ("heldout", "all"):
        entry = manifest["files"]["heldout"]
        if Path(entry["path"]).name != entry["path"]:
            raise ValueError("保留集路径须位于同一资料目录")
        reserved = _verified_json(path.with_name(entry["path"]), entry["sha256"])
        queries.extend(reserved["queries"])
    _validate(main["candidates"], queries)
    return {"candidates": main["candidates"], "queries": queries,
            "metadata": {"dataset_revision": manifest["revision"], "frozen_at": manifest["frozen_at"],
                         "split": split, "files": manifest["files"],
                         "corpus_sha256": _digest(main["candidates"]),
                         "queries_sha256": _digest(queries),
                         "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}}


def _ratio(total, count):
    return total / count if count else None


def _metrics(rows, k):
    positive = [row for row in rows if row["label"] == "supported"]
    strict = [row for row in rows if row["label"] != "supported" and row["retrieval_expectation"] != "background_allowed"]
    irrelevant = [row for row in strict if row["label"] == "irrelevant"]
    unsupported = [row for row in strict if row["label"] == "unsupported"]
    background = [row for row in rows if row["retrieval_expectation"] == "background_allowed"]
    returned = lambda items: sum(bool(row["retrieved_reference_ids"]) for row in items)
    return {"positive_queries": len(positive),
            f"recall_at_{k}": _ratio(sum(row["support_recall"] for row in positive), len(positive)),
            f"hit_at_{k}": _ratio(sum(row["first_support_rank"] is not None for row in positive), len(positive)),
            f"mrr_at_{k}": _ratio(sum(row["reciprocal_rank"] for row in positive), len(positive)),
            "top1": _ratio(sum(row["first_support_rank"] == 1 for row in positive), len(positive)),
            "negative_queries": len(strict), "negative_returned_queries": returned(strict),
            "negative_false_return_rate": _ratio(returned(strict), len(strict)),
            "irrelevant_queries": len(irrelevant), "irrelevant_false_return_rate": _ratio(returned(irrelevant), len(irrelevant)),
            "unsupported_queries": len(unsupported), "unsupported_return_rate": _ratio(returned(unsupported), len(unsupported)),
            "background_probes": len(background), "background_probe_return_rate": _ratio(returned(background), len(background)),
            "background_match_rate": _ratio(sum(bool(row["returned_background_reference_ids"]) for row in background), len(background))}


def _observed_retrieval(retrieve_fn, question, candidates, embedding, k):
    errors, calls = [], []

    def tracked_embedding(texts):
        calls.append(True)
        try:
            return embedding(texts)
        except Exception as error:
            errors.append("error:" + type(error).__name__)
            raise

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", RuntimeWarning)
        result = retrieve_fn(question, candidates,
                             embedding=tracked_embedding if callable(embedding) else embedding, limit=k)
    entries = result if isinstance(result, list) else []
    modes = {row["retrieval_mode"] for row in entries if isinstance(row, dict) and row.get("retrieval_mode")}
    statuses = {row["embedding_status"] for row in entries if isinstance(row, dict) and row.get("embedding_status")}
    vector_warnings = [str(item.message) for item in emitted
                       if issubclass(item.category, RuntimeWarning) and "检索向量" in str(item.message)]
    fallback = bool(errors or vector_warnings or "lexical_fallback" in modes)
    if fallback:
        mode = "lexical_fallback"
        status = (errors[-1] if errors else "invalid_vectors" if "invalid_vectors" in statuses
                  or any("格式" in message for message in vector_warnings)
                  else next(iter(sorted(statuses)), "unavailable"))
    elif len(modes) > 1 or len(statuses) > 1:
        mode, status = "mixed", "mixed"
    else:
        mode = next(iter(modes), "hybrid" if calls else "lexical" if embedding is None else "unspecified")
        status = next(iter(statuses), "available" if calls else "not_configured" if embedding is None else "not_reported")
    valid = not fallback and status not in ("mixed", "invalid_vectors", "unavailable")
    return result, mode, status, valid


def evaluate_benchmark(candidates, queries, retrieve_fn, embedding=None, k=5) -> dict:
    """Score supporting passages, keeping unverified-field-fact probes separate."""
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50:
        raise ValueError("评测k必须为1至50的整数")
    _validate(candidates, queries)
    references = {row["reference_id"] for row in candidates}
    rows = []
    for query in queries:
        started = time.perf_counter()
        result, mode, status, valid = _observed_retrieval(retrieve_fn, query["query"], candidates, embedding, k)
        if (not isinstance(result, list) or any(not isinstance(row, dict) or row.get("reference_id") not in references for row in result)
                or len({row["reference_id"] for row in result}) != len(result)):
            raise ValueError("评测检索结果含未知、重复或无效引用")
        result = result[:k]
        ids = [item["reference_id"] for item in result]
        gold = set(query["supporting_reference_ids"])
        ranks = [index for index, reference in enumerate(ids, 1) if reference in gold]
        first = min(ranks) if ranks else None
        background = set(query.get("background_relevant_reference_ids", []))
        rows.append({"query_id": query["query_id"], "topic_id": query["topic_id"], "split": query["split"],
                     "query": query["query"], "label": query["label"],
                     "retrieval_expectation": query.get("retrieval_expectation",
                                                       "supporting_passage" if query["label"] == "supported" else "no_supporting_passage"),
                     "retrieval_mode": mode, "embedding_status": status, "valid": valid,
                     "supporting_reference_ids": sorted(gold), "retrieved_reference_ids": ids,
                     "returned_background_reference_ids": sorted(background.intersection(ids)),
                     "returned_hard_negative_reference_ids": sorted(set(query.get("hard_negative_reference_ids", [])).intersection(ids)),
                     "first_support_rank": first, "support_recall": len(gold.intersection(ids)) / len(gold) if gold else None,
                     "reciprocal_rank": 1 / first if first else 0,
                     "duration_seconds": round(time.perf_counter() - started, 6),
                     "ranking": [{key: item[key] for key in ("reference_id", "score", "lexical_score", "semantic_score",
                                                           "rerank_reasons", "retrieval_mode", "embedding_status")
                                  if key in item} for item in result]})
    valid_rows = [row for row in rows if row["valid"]]
    invalid_ids = [row["query_id"] for row in rows if not row["valid"]]
    return {"k": k, "metrics": _metrics(valid_rows, k), "metrics_all_queries": _metrics(rows, k), "per_query": rows,
            "validity": {"valid": not invalid_ids, "invalid_query_ids": invalid_ids,
                         "metrics_scope": "valid_queries_only", "total_queries": len(rows), "valid_queries": len(valid_rows)},
            "diagnostics": {"fallback_queries": sum(row["retrieval_mode"] == "lexical_fallback" for row in rows),
                            "retrieval_mode_counts": dict(Counter(row["retrieval_mode"] for row in rows)),
                            "embedding_status_counts": dict(Counter(row["embedding_status"] for row in rows))},
            "by_split": {split: _metrics([row for row in valid_rows if row["split"] == split], k) for split in sorted({row["split"] for row in rows})},
            "by_topic": {topic: _metrics([row for row in valid_rows if row["topic_id"] == topic], k) for topic in sorted({row["topic_id"] for row in rows})},
            "definitions": {"recall": "Mean fraction of gold supporting passages present in top-k, over supported questions only.",
                            "mrr": "Mean reciprocal first supporting rank within top-k, over supported questions only.",
                            "negative_false_return": "Any returned candidate for strict irrelevant/unsupported questions; measures retrieval candidates, not fabricated generation.",
                            "background_probes": "Relevant background may be retrieved but does not prove the requested field fact; reported separately.",
                            "validity": "Semantic fallback or mixed status invalidates the run; primary metrics exclude invalid queries, all-query metrics remain diagnostic."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("dev", "heldout", "all"), default="dev")
    parser.add_argument("--k", type=int, choices=(5, 10, 50), default=5)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-revision", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime = output.parent / "runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(runtime / "hf")
    os.environ["TMP"] = os.environ["TEMP"] = str(runtime)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from app.adapters.text_embeddings import LocalTextEmbeddings
    from app.services.knowledge.retrieval import retrieve
    loaded = load_benchmark(args.dataset, split=args.split)
    attempted_network = []
    def blocked(*args, **kwargs):
        attempted_network.append(True)
        raise RuntimeError("benchmark network access is disabled")
    with patch.object(socket, "create_connection", blocked), patch.object(socket.socket, "connect", blocked):
        encoder = LocalTextEmbeddings(args.model_dir, args.index_dir)
        encoder([row["title"] + "\n" + row["text"] for row in loaded["candidates"]])
        warmup = dict(encoder.last_run)
        report = evaluate_benchmark(loaded["candidates"], loaded["queries"], retrieve, embedding=encoder, k=args.k)
    manifest = json.loads((Path(args.model_dir) / "manifest.json").read_text(encoding="utf-8"))
    report.update(schema_version="knowledge-retrieval-evaluation-v1", evaluated_at=datetime.now(timezone.utc).isoformat(),
                  dataset=loaded["metadata"], system={"revision": args.system_revision, "python": platform.python_version(),
                  "retrieval_source": inspect.getsourcefile(retrieve),
                  "retrieval_sha256": hashlib.sha256(Path(inspect.getsourcefile(retrieve)).read_bytes()).hexdigest(),
                  "embedding_sha256": hashlib.sha256(Path(inspect.getsourcefile(LocalTextEmbeddings)).read_bytes()).hexdigest(),
                  "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  model={"repo_id": manifest["repo_id"], "revision": manifest["revision"], "model_key": encoder.model_key,
                         "onnx_sha256": manifest["files"]["model.onnx"]["sha256"]},
                  cache_warmup=warmup, network_attempts=len(attempted_network), cloud_calls=0)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "metrics": report["metrics"], "by_split": report["by_split"],
                      "network_attempts": len(attempted_network), "cloud_calls": 0}, ensure_ascii=True))
    return 1 if attempted_network or not report["validity"]["valid"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
