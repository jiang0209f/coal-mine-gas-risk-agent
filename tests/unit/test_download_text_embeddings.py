"""Official model acquisition uses pinned revisions and verified local artifacts."""
from copy import deepcopy
import hashlib
from importlib import import_module
import json

import httpx
import pytest


@pytest.fixture
def module():
    return import_module("app.scripts.download_text_embeddings")


def public_repo(module):
    files = {filename: ("synthetic " + filename).encode() for filename in module.MODEL_FILES}
    files["config.json"] = b'{"model_type":"bert","hidden_size":3,"max_position_embeddings":512}'
    siblings = []
    for name, body in files.items():
        record = {"rfilename": name, "size": len(body),
                  "blobId": hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()}
        if name == "model.safetensors":
            record["lfs"] = {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
        siblings.append(record)
    return files, {"sha": module.MODEL_REVISION, "siblings": siblings}


def client_for(module, files, metadata, calls):
    def respond(request):
        calls.append(str(request.url))
        if request.url.path.startswith("/api/models/"):
            return httpx.Response(200, json=metadata)
        marker = "/resolve/" + module.MODEL_REVISION + "/"
        assert marker in request.url.path
        filename = request.url.path.split(marker, 1)[1]
        return httpx.Response(200, content=files[filename])
    return httpx.Client(transport=httpx.MockTransport(respond))


def test_pinned_official_files_are_saved_with_verified_sha_manifest(module, tmp_path):
    files, metadata = public_repo(module)
    calls = []
    with client_for(module, files, metadata, calls) as client:
        manifest = module.download_model(tmp_path / "model", client=client)
    assert manifest["repo_id"] == "BAAI/bge-small-zh-v1.5"
    assert manifest["revision"] == module.MODEL_REVISION
    assert set(manifest["files"]) == set(files)
    assert all("pytorch_model.bin" not in call for call in calls)
    for filename, body in files.items():
        entry = manifest["files"][filename]
        assert (tmp_path / "model" / filename).read_bytes() == body
        assert entry["sha256"] == hashlib.sha256(body).hexdigest()
        assert entry["bytes"] == len(body)
        assert module.MODEL_REVISION in entry["source_url"]
    assert json.loads((tmp_path / "model" / "manifest.json").read_text()) == manifest


def test_existing_verified_files_are_reused_without_redownload(module, tmp_path):
    files, metadata = public_repo(module)
    with client_for(module, files, metadata, []) as client:
        module.download_model(tmp_path, client=client)
    calls = []
    with client_for(module, files, metadata, calls) as client:
        manifest = module.download_model(tmp_path, client=client)
    assert len(calls) == 1
    assert manifest["files"]["model.safetensors"]["sha256"] == hashlib.sha256(files["model.safetensors"]).hexdigest()


def test_corrupt_existing_weight_is_rejected_without_overwriting_manifest(module, tmp_path):
    files, metadata = public_repo(module)
    with client_for(module, files, metadata, []) as client:
        module.download_model(tmp_path, client=client)
    previous = (tmp_path / "manifest.json").read_bytes()
    (tmp_path / "model.safetensors").write_bytes(b"damaged")
    with client_for(module, files, metadata, []) as client:
        with pytest.raises(ValueError, match="已有.*校验"):
            module.download_model(tmp_path, client=client)
    assert (tmp_path / "model.safetensors").read_bytes() == b"damaged"
    assert (tmp_path / "manifest.json").read_bytes() == previous


@pytest.mark.parametrize("filename", ["model.safetensors", "config.json"])
def test_remote_lfs_or_git_blob_mismatch_is_rejected_and_partial_file_removed(module, tmp_path, filename):
    files, metadata = public_repo(module)
    served = {**files, filename: b"?" * len(files[filename])}
    with client_for(module, served, metadata, []) as client:
        with pytest.raises(ValueError, match="校验"):
            module.download_model(tmp_path, client=client)
    assert not (tmp_path / filename).exists()
    assert not list(tmp_path.rglob("*.part"))
    assert not (tmp_path / "manifest.json").exists()


def test_metadata_revision_mismatch_stops_before_file_download(module, tmp_path):
    files, metadata = public_repo(module)
    metadata["sha"] = "0" * 40
    calls = []
    with client_for(module, files, metadata, calls) as client:
        with pytest.raises(ValueError, match="revision"):
            module.download_model(tmp_path, client=client)
    assert len(calls) == 1 and not (tmp_path / "model.safetensors").exists()


def test_missing_required_official_file_is_rejected(module, tmp_path):
    files, metadata = public_repo(module)
    metadata["siblings"] = [entry for entry in metadata["siblings"] if entry["rfilename"] != "model.safetensors"]
    with client_for(module, files, metadata, []) as client:
        with pytest.raises(ValueError, match="缺少"):
            module.download_model(tmp_path, client=client)


@pytest.mark.parametrize("revision", ["main", "refs/pr/1", "../model", "f" * 39])
def test_branch_or_invalid_revision_is_rejected_before_any_network_call(module, tmp_path, revision):
    files, metadata = public_repo(module)
    calls = []
    with client_for(module, files, metadata, calls) as client:
        with pytest.raises(ValueError, match="revision"):
            module.download_model(tmp_path, revision=revision, client=client)
    assert calls == []


def test_export_verifies_original_weight_before_importing_model_dependencies(module, tmp_path):
    files, metadata = public_repo(module)
    with client_for(module, files, metadata, []) as client:
        module.download_model(tmp_path, client=client)
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="校验"):
        module.export_model(tmp_path)


def test_export_rejects_manifest_path_escape_before_reading_external_file(module, tmp_path):
    files, metadata = public_repo(module)
    with client_for(module, files, metadata, []) as client:
        module.download_model(tmp_path, client=client)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["../outside.safetensors"] = deepcopy(manifest["files"]["model.safetensors"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="路径"):
        module.export_model(tmp_path)
