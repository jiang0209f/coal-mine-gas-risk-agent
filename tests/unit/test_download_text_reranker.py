"""Pinned public asset acquisition, using only synthetic HTTP responses."""

import copy
import hashlib
from importlib import import_module
import json

import httpx
import pytest


@pytest.fixture
def module():
    return import_module("app.scripts.download_text_reranker")


def manifest(module):
    files = {name: {"sha256": "a" * 64, "bytes": 1} for name in module.FILES}
    files[module.MODEL_FILE] = {"sha256": module.MODEL_SHA, "bytes": 118620016}
    files["LICENSE.apache-2.0.txt"] = {"sha256": "b" * 64, "bytes": 11358}
    return {"repo_id": module.REPO, "revision": module.REVISION,
            "model_file": module.MODEL_FILE, "license": "apache-2.0", "files": files}


def test_valid_fixed_identity_is_accepted(module):
    module.verify_manifest_identity(manifest(module))


@pytest.mark.parametrize("field", ["repo_id", "revision", "model_file", "license"])
def test_wrong_fixed_identity_is_rejected(module, field):
    value = manifest(module)
    value[field] = "wrong"
    with pytest.raises(ValueError):
        module.verify_manifest_identity(value)


@pytest.mark.parametrize("case", ["empty", "missing", "unexpected", "model_digest", "model_size"])
def test_incomplete_or_wrong_files_are_rejected(module, case):
    value = copy.deepcopy(manifest(module))
    if case == "empty":
        value["files"] = {}
    elif case == "missing":
        value["files"].pop("tokenizer.json")
    elif case == "unexpected":
        value["files"]["../../other.onnx"] = {"sha256": "a" * 64, "bytes": 1}
    elif case == "model_digest":
        value["files"][module.MODEL_FILE]["sha256"] = "f" * 64
    else:
        value["files"][module.MODEL_FILE]["bytes"] = 1
    with pytest.raises(ValueError):
        module.verify_manifest_identity(value)


@pytest.fixture
def server(module, monkeypatch):
    assets = {name: ("synthetic " + name).encode() for name in module.FILES}
    assets["config.json"] = json.dumps({"architectures": ["XLMRobertaForSequenceClassification"]}).encode()
    monkeypatch.setattr(module, "MODEL_SHA", hashlib.sha256(assets[module.MODEL_FILE]).hexdigest())
    monkeypatch.setattr(module, "MODEL_BYTES", len(assets[module.MODEL_FILE]))
    siblings = []
    for name, body in assets.items():
        entry = {"rfilename": name, "size": len(body),
                 "blobId": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()}
        if name == module.MODEL_FILE:
            entry["lfs"] = {"sha256": module.MODEL_SHA, "size": len(body)}
        siblings.append(entry)
    metadata = {"sha": module.REVISION, "cardData": {"license": "apache-2.0"}, "siblings": siblings}
    requests, overrides = [], {}
    license_body = b"Apache License\nVersion 2.0, January 2004\nsynthetic license fixture"

    def handler(request):
        requests.append(str(request.url))
        if request.url.path.startswith("/api/models/"):
            assert request.url.params["blobs"] == "true"
            return httpx.Response(200, json=metadata)
        if request.url.host == "www.apache.org":
            return httpx.Response(200, content=overrides.get("license", license_body))
        prefix = f"/{module.REPO}/resolve/{module.REVISION}/"
        if request.url.path.startswith(prefix):
            name = request.url.path[len(prefix):]
            body = overrides.get(name, assets.get(name))
            if isinstance(body, httpx.SyncByteStream):
                return httpx.Response(200, stream=body)
            if body is not None:
                return httpx.Response(200, content=body)
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        yield {"client": client, "assets": assets, "metadata": metadata,
               "requests": requests, "overrides": overrides, "handler": handler}


def test_downloads_only_allowlisted_files_with_verified_manifest(module, server, tmp_path):
    server["metadata"]["siblings"].append({"rfilename": "../../other-model.onnx"})
    result = module.download_model(tmp_path, client=server["client"])
    module.verify_manifest_identity(result)
    assert set(result["files"]) == {*module.FILES, "LICENSE.apache-2.0.txt"}
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == result
    for name, entry in result["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == entry["sha256"]
    assert not any("other-model" in url for url in server["requests"])
    assert result["trust_remote_code"] is False
    assert result["architecture"] == "XLMRobertaForSequenceClassification"
    assert server["client"].is_closed is False


def test_reuses_verified_originals_without_downloading_model_again(module, server, tmp_path):
    module.download_model(tmp_path, client=server["client"])
    server["requests"].clear()
    module.download_model(tmp_path, client=server["client"])
    assert not any("/resolve/" in url for url in server["requests"])


def test_existing_bad_original_is_preserved_and_rejected(module, server, tmp_path):
    target = tmp_path / module.MODEL_FILE
    target.parent.mkdir()
    target.write_bytes(b"bad existing file")
    with pytest.raises(ValueError):
        module.download_model(tmp_path, client=server["client"])
    assert target.read_bytes() == b"bad existing file"
    assert not any("/resolve/" in url for url in server["requests"])
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("case", ["revision", "license", "model_digest", "model_size", "missing"])
def test_wrong_remote_identity_is_rejected_before_asset_download(module, server, tmp_path, case):
    metadata = server["metadata"]
    if case == "revision":
        metadata["sha"] = "b" * 40
    elif case == "license":
        metadata["cardData"]["license"] = "unknown"
    elif case == "model_digest":
        metadata["siblings"][0]["lfs"]["sha256"] = "f" * 64
    elif case == "model_size":
        metadata["siblings"][0]["lfs"]["size"] += 1
    else:
        metadata["siblings"].pop()
    with pytest.raises(ValueError):
        module.download_model(tmp_path, client=server["client"])
    assert not any("/resolve/" in url for url in server["requests"])


@pytest.mark.parametrize("kind", ["same_size_bad_digest", "short", "oversized", "interrupted"])
def test_failed_download_removes_partial_file_and_writes_no_manifest(module, server, tmp_path, kind):
    if kind == "interrupted":
        class Broken(httpx.SyncByteStream):
            def __iter__(self):
                yield b"part"
                raise httpx.ReadError("synthetic interrupted transfer")
        server["overrides"][module.MODEL_FILE] = Broken()
    else:
        count = len(server["assets"][module.MODEL_FILE]) + {"short": -1, "oversized": 1}.get(kind, 0)
        server["overrides"][module.MODEL_FILE] = b"x" * count
    with pytest.raises((ValueError, httpx.ReadError)):
        module.download_model(tmp_path, client=server["client"])
    assert not (tmp_path / module.MODEL_FILE).exists()
    assert list(tmp_path.rglob("*.part")) == []
    assert not (tmp_path / "manifest.json").exists()


def test_non_lfs_asset_digest_is_also_verified(module, server, tmp_path):
    original = server["assets"]["tokenizer.json"]
    server["overrides"]["tokenizer.json"] = b"x" * len(original)
    with pytest.raises(ValueError):
        module.download_model(tmp_path, client=server["client"])
    assert not (tmp_path / "tokenizer.json").exists()
    assert not (tmp_path / "manifest.json").exists()


def test_license_failure_does_not_publish_a_manifest(module, server, tmp_path):
    server["overrides"]["license"] = b"HTML login page"
    with pytest.raises(ValueError):
        module.download_model(tmp_path, client=server["client"])
    assert not (tmp_path / "manifest.json").exists()


def test_cli_requires_an_explicit_nonempty_model_directory(module, monkeypatch):
    monkeypatch.setattr(module, "download_model", lambda *a, **k: pytest.fail("unexpected download"))
    for argv in ([], ["--model-dir", " "]):
        with pytest.raises(SystemExit) as caught:
            module.main(argv)
        assert caught.value.code == 2


def test_cli_writes_optional_report_and_small_stdout_metadata(module, monkeypatch, tmp_path, capsys):
    result = manifest(module)
    result["download_bytes"] = 123
    calls = []
    def download(path):
        calls.append(path)
        return result
    monkeypatch.setattr(module, "download_model", download)
    model_dir, report_dir = tmp_path / "model", tmp_path / "report"
    assert module.main(["--model-dir", str(model_dir), "--report-dir", str(report_dir)]) == 0
    assert calls == [model_dir]
    assert json.loads((report_dir / "download-verification.json").read_text(encoding="utf-8")) == result
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["status"] == "verified" and stdout["revision"] == module.REVISION
    assert "files" not in stdout


@pytest.mark.parametrize("card", [None, []])
def test_malformed_remote_license_metadata_is_a_validation_failure(module, server, tmp_path, card):
    server["metadata"]["cardData"] = card
    with pytest.raises(ValueError):
        module.download_model(tmp_path, client=server["client"])
    assert not any("/resolve/" in url for url in server["requests"])


def test_owned_http_client_disables_environment_credentials_and_is_closed(module, server, monkeypatch, tmp_path):
    options, created = [], []
    client_class = httpx.Client
    def create(**kwargs):
        options.append(kwargs)
        client = client_class(transport=httpx.MockTransport(server["handler"]), **kwargs)
        created.append(client)
        return client
    monkeypatch.setattr(module.httpx, "Client", create)
    module.download_model(tmp_path)
    assert options == [{"trust_env": False, "follow_redirects": True, "timeout": 120}]
    assert created[0].is_closed is True
