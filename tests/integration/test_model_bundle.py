"""Local model bundles validate identity, content hashes and file boundaries."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from app.adapters.model_bundle import ModelBundle, ModelNotReady


def write_manifest(root: Path, **changes):
    root.mkdir(parents=True, exist_ok=True)
    value = {
        "kind": "gas_attribution", "model_id": "fixture-attribution", "version": "v1",
        "artifacts": {},
    }
    value.update(changes)
    (root / "manifest.json").write_text(json.dumps(value), encoding="utf-8")
    return value


def artifact(root: Path, name="classifier.skops", content=b"synthetic test artifact"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}


def require_bundle(root, names=("classifier",)):
    return ModelBundle(root, "gas_attribution").require_artifacts(names)


def test_absent_manifest_is_not_ready(tmp_path):
    with pytest.raises(ModelNotReady) as caught:
        ModelBundle(tmp_path / "absent", "gas_attribution")

    assert caught.value.missing_fields


@pytest.mark.parametrize("text", ["{", "[]", "null", "true", "7", '"manifest"'])
def test_broken_or_non_object_manifest_is_rejected(tmp_path, text):
    (tmp_path / "manifest.json").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        ModelBundle(tmp_path, "gas_attribution")


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_are_rejected(tmp_path, constant):
    manifest = write_manifest(tmp_path)
    text = json.dumps(manifest)[:-1] + ', "threshold": ' + constant + "}"
    (tmp_path / "manifest.json").write_text(text, encoding="utf-8")

    with pytest.raises(ValueError):
        ModelBundle(tmp_path, "gas_attribution")


@pytest.mark.parametrize("change", [
    {"kind": "gas_forecast"}, {"kind": ""}, {"kind": None},
    {"model_id": ""}, {"model_id": "   "}, {"model_id": 3},
    {"version": ""}, {"version": None}, {"version": True},
])
def test_bundle_identity_must_match_expected_kind_and_nonempty_strings(tmp_path, change):
    write_manifest(tmp_path, **change)

    with pytest.raises(ValueError):
        ModelBundle(tmp_path, "gas_attribution")


def test_valid_bundle_returns_resolved_files_without_changing_manifest(tmp_path):
    descriptor = artifact(tmp_path)
    manifest = write_manifest(tmp_path, artifacts={"classifier": descriptor})
    before = deepcopy(manifest)
    bundle = ModelBundle(tmp_path, "gas_attribution")

    paths = bundle.require_artifacts(["classifier"])

    assert bundle.root == tmp_path.resolve()
    assert bundle.manifest == before
    assert paths == {"classifier": (tmp_path / descriptor["path"]).resolve()}
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == before


def test_nested_relative_artifact_within_bundle_is_accepted(tmp_path):
    descriptor = artifact(tmp_path, "models/classifier.skops")
    write_manifest(tmp_path, artifacts={"classifier": descriptor})

    paths = require_bundle(tmp_path)

    assert paths["classifier"] == (tmp_path / "models/classifier.skops").resolve()


def test_hash_mismatch_is_rejected_before_loading_model(tmp_path):
    descriptor = artifact(tmp_path)
    write_manifest(tmp_path, artifacts={"classifier": descriptor})
    (tmp_path / descriptor["path"]).write_bytes(b"replacement bytes")

    with pytest.raises(ValueError):
        require_bundle(tmp_path)


def test_each_require_rechecks_file_after_a_previous_success(tmp_path):
    descriptor = artifact(tmp_path)
    write_manifest(tmp_path, artifacts={"classifier": descriptor})
    bundle = ModelBundle(tmp_path, "gas_attribution")
    bundle.require_artifacts(["classifier"])
    (tmp_path / descriptor["path"]).write_bytes(b"changed after first verification")

    with pytest.raises(ValueError):
        bundle.require_artifacts(["classifier"])


@pytest.mark.parametrize("declared", [False, True])
def test_absent_artifact_is_not_ready_with_artifact_name(tmp_path, declared):
    descriptor = {"path": "classifier.skops", "sha256": "0" * 64}
    write_manifest(tmp_path, artifacts={"classifier": descriptor} if declared else {})

    with pytest.raises(ModelNotReady) as caught:
        require_bundle(tmp_path)

    assert any("classifier" in field for field in caught.value.missing_fields)


@pytest.mark.parametrize("outside_path", ["../outside.txt", "..\\outside.txt"])
def test_parent_traversal_is_rejected_even_with_correct_hash(tmp_path, outside_path):
    root = tmp_path / "bundle"
    root.mkdir()
    content = b"outside the bundle"
    (tmp_path / "outside.txt").write_bytes(content)
    write_manifest(root, artifacts={"classifier": {
        "path": outside_path, "sha256": hashlib.sha256(content).hexdigest(),
    }})

    with pytest.raises(ValueError):
        require_bundle(root)


def test_absolute_artifact_path_is_rejected_even_when_file_is_inside_bundle(tmp_path):
    descriptor = artifact(tmp_path)
    descriptor["path"] = str((tmp_path / descriptor["path"]).resolve())
    write_manifest(tmp_path, artifacts={"classifier": descriptor})

    with pytest.raises(ValueError):
        require_bundle(tmp_path)


def test_symlink_cannot_escape_bundle(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"external model")
    link = root / "classifier.skops"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlink creation unavailable: {error}")
    write_manifest(root, artifacts={"classifier": {
        "path": link.name, "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
    }})

    with pytest.raises(ValueError):
        require_bundle(root)


@pytest.mark.parametrize("artifacts", [
    None, [], "classifier.skops", {"classifier": None},
    {"classifier": "classifier.skops"},
    {"classifier": {"path": None, "sha256": "0" * 64}},
    {"classifier": {"path": "classifier.skops", "sha256": "bad-hash"}},
    {"classifier": {"path": "classifier.skops", "sha256": "g" * 64}},
    {"classifier": {"path": "classifier.skops", "sha256": 64}},
])
def test_malformed_artifact_declarations_are_rejected(tmp_path, artifacts):
    write_manifest(tmp_path, artifacts=artifacts)

    with pytest.raises(ValueError):
        require_bundle(tmp_path)


@pytest.mark.parametrize("extension", [".pkl", ".pickle", ".joblib"])
def test_pickle_based_artifact_formats_are_rejected(tmp_path, extension):
    descriptor = artifact(tmp_path, "classifier" + extension)
    write_manifest(tmp_path, artifacts={"classifier": descriptor})

    with pytest.raises(ValueError):
        require_bundle(tmp_path)


@pytest.mark.parametrize("extension", [".json", ".skops", ".pt", ".pth", ".txt", ".npz"])
def test_supported_file_formats_are_verified_without_deserialization(tmp_path, extension):
    descriptor = artifact(tmp_path, "weights" + extension)
    write_manifest(tmp_path, artifacts={"classifier": descriptor})

    paths = require_bundle(tmp_path)

    assert paths["classifier"].read_bytes() == b"synthetic test artifact"
