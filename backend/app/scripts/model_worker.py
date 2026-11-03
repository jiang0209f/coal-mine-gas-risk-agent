"""JSON-only, isolated entry point for the two professional model wrappers."""

import argparse
from contextlib import redirect_stdout
import io
import json
import sys

from app.adapters.model_bundle import ModelNotReady


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地专业模型推理")
    parser.add_argument("--kind", choices=("gas_forecast", "gas_attribution"), required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--inspect", action="store_true")
    arguments = parser.parse_args(argv)
    result = {}
    adapter = None
    try:
        with redirect_stdout(io.StringIO()):
            if arguments.kind == "gas_forecast":
                from app.adapters.gas_forecast import GasForecastAdapter
                adapter = GasForecastAdapter(arguments.bundle)
            else:
                from app.adapters.gas_attribution import GasAttributionAdapter
                adapter = GasAttributionAdapter(arguments.bundle)
            if arguments.inspect:
                result = adapter.inspect()
            else:
                content = sys.stdin.read(8 * 1024 * 1024 + 1)
                if len(content) > 8 * 1024 * 1024:
                    raise ValueError("输入超过限制")
                request = json.loads(content, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("非有限JSON")))
                if not isinstance(request, dict):
                    raise ValueError("模型请求须为对象")
                result = adapter.predict(request)
    except ModelNotReady as error:
        result = {"status": "not_ready", "missing_fields": error.missing_fields}
    except (ValueError, KeyError, TypeError):
        result = {"status": "failed", "error_code": "invalid_model_input_or_package"}
    except Exception:
        result = {"status": "failed", "error_code": "model_inference_failed"}
    if adapter is not None:
        result.setdefault("model_id", adapter.manifest["model_id"])
        result.setdefault("model_version", adapter.manifest["version"])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
