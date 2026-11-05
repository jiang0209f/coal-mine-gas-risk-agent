"""Record outgoing field names and a request digest without storing content or credentials."""
import hashlib
import json
from uuid import uuid4
from app.schemas.common import now_utc, utcstr


def field_paths(value, prefix=""):
    if isinstance(value,dict):
        return [path for key,child in value.items() for path in field_paths(child,f"{prefix}.{key}".lstrip("."))]
    if isinstance(value,list):
        return [path for index,child in enumerate(value) for path in field_paths(child,f"{prefix}[{index}]" )]
    if isinstance(value,str) and prefix.endswith(".content"):
        try:
            child=json.loads(value)
            if isinstance(child,(dict,list)): return field_paths(child,prefix+".json")
        except (ValueError,TypeError): pass
    return [prefix]


def record_disclosure(store, task, messages, tools):
    payload={"messages":messages,"tools":tools}
    identifier=str(uuid4())
    store.put("cloud_disclosure",identifier,dict(disclosure_id=identifier,
        task_id=task.get("task_id",task.get("report_id")),source_id=task["source_id"],session_id=task["session_id"],
        fields=sorted(set(field_paths(payload))),sha256=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
        endpoint="https://api.deepseek.com/chat/completions",created_at=utcstr(now_utc())))
    return identifier
