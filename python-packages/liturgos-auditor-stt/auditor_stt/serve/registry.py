"""Model registry on the filesystem: which fine-tuned models exist and which one is current.

Layout (no symlinks, so it works on Windows and in Docker volumes alike):

    <registry_dir>/<name>/<version>/ct2/            the CTranslate2 export faster-whisper loads
    <registry_dir>/<name>/<version>/metadata.json   lineage: dataset version, base model, gate result
    <registry_dir>/<name>/<version>/gate.json       the evaluation gate's verdict (if there is one)
    <registry_dir>/<name>/current.json              the promoted version, written atomically

`promote` is the only way `current.json` changes, and it refuses a version whose
gate is missing or failed unless forced (the force is recorded). The service
and the CLI both read this module; it needs nothing beyond the standard library
so the small serving image can import it.

Names and versions become directory names, so they are matched against a strict
pattern before any path is built: a spec can never reach outside the registry.
"""

import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_DIR = "./models/registry"
REGISTRY_PREFIX = "registry:"

CT2_DIRNAME = "ct2"
METADATA_FILENAME = "metadata.json"
GATE_FILENAME = "gate.json"
CURRENT_FILENAME = "current.json"
TRAINING_METADATA_FILENAME = "training_metadata.json"
# `export --merge-lora` leaves its ~3 GB fp32 merged checkpoint inside the export directory;
# it is not something the service loads and never belongs in the registry.
MERGED_HF_DIRNAME = "_merged_hf"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_LARGE_TRAINING_FIELD_BYTES = 2000  # training metadata values bigger than this are not copied


class RegistryError(Exception):
    """The registry operation cannot be done (bad name, unknown model, existing version, ...)."""


class GateRefusedError(RegistryError):
    """A version was not promoted because its gate is missing or did not pass."""


def default_registry_dir():
    return Path(os.environ.get("AUDITOR_STT_REGISTRY_DIR") or DEFAULT_REGISTRY_DIR)


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_name(kind, value):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise RegistryError(
            f"Invalid {kind} {value!r}: use 1-64 letters, digits, '.', '_' or '-', starting with a letter or digit"
        )
    return value


def _model_dir(registry_dir, name):
    return Path(registry_dir) / _check_name("model name", name)


def _version_dir(registry_dir, name, version):
    return _model_dir(registry_dir, name) / _check_name("version", version)


def _read_json(path):
    """The parsed JSON object in `path`, or None when it is missing, unreadable or not an object."""
    try:
        content = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return content if isinstance(content, dict) else None


def _temp_name(path):
    return path.with_name(f".{path.name}-{uuid.uuid4().hex[:8]}.tmp")


def _write_json_atomic(path, content):
    # Not tempfile.mkstemp/mkdtemp: they create private (0600/0700) entries, and the registry
    # is often a volume shared with a service container running as another user.
    path = Path(path)
    tmp = _temp_name(path)
    try:
        with open(tmp, "x", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --- register --------------------------------------------------------------


def register(
    registry_dir,
    name,
    version,
    export_dir,
    *,
    training_run_dir=None,
    gate_path=None,
    quantization=None,
    move=False,
):
    """Add a CTranslate2 export to the registry as `<name>/<version>` and return its metadata.

    The export is copied without `_merged_hf` and `gate.json`; the gate is
    stored next to the metadata instead, taken from `gate_path` or else from
    the export directory. An existing version is never overwritten. The version
    is assembled in a hidden temporary directory and renamed into place, so a
    failed registration leaves nothing behind. With `move` the source files
    are removed only after the version is in place, so a failure never loses
    the export (at the price of needing room for both copies meanwhile).
    """
    export_dir = Path(export_dir)
    target = _version_dir(registry_dir, name, version)
    if target.exists():
        raise RegistryError(f"{name}@{version} is already registered; versions are never overwritten")
    if not export_dir.is_dir():
        raise RegistryError(f"Export directory {export_dir} does not exist")
    if not (export_dir / "model.bin").is_file():
        raise RegistryError(f"{export_dir} is not a CTranslate2 export (no model.bin)")

    gate_source = Path(gate_path) if gate_path is not None else export_dir / GATE_FILENAME
    if gate_path is not None and not gate_source.is_file():
        raise RegistryError(f"Gate file {gate_source} does not exist")
    gate = _read_json(gate_source) if gate_source.is_file() else None
    if gate_path is not None and gate is None:
        raise RegistryError(f"Gate file {gate_source} is not a readable JSON object")

    training = None
    if training_run_dir is not None:
        training = _read_json(Path(training_run_dir) / TRAINING_METADATA_FILENAME)
        if training is None:
            raise RegistryError(f"{Path(training_run_dir) / TRAINING_METADATA_FILENAME} is missing or unreadable")

    metadata = _metadata(name, version, export_dir, gate, training, training_run_dir, quantization)

    model_dir = target.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    staging = _temp_name(target)
    staging.mkdir()
    try:
        shutil.copytree(export_dir, staging / CT2_DIRNAME, ignore=_excluded_from_export(export_dir))
        if gate is not None:
            shutil.copyfile(gate_source, staging / GATE_FILENAME)
        _write_json_atomic(staging / METADATA_FILENAME, metadata)
        try:
            os.rename(staging, target)
        except OSError as exc:
            raise RegistryError(f"Could not create {name}@{version}: {exc}") from exc
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if move:
        _remove_registered_files(export_dir, gate_source if gate_path is None and gate is not None else None)
    return metadata


def _excluded_from_export(export_dir):
    def ignore(directory, names):
        if Path(directory) != export_dir:
            return ()
        return {n for n in names if n in (MERGED_HF_DIRNAME, GATE_FILENAME)}

    return ignore


def _remove_registered_files(export_dir, consumed_gate):
    """After a `move`: drop what was copied, keep `_merged_hf`, and remove the directory if nothing is left."""
    for entry in export_dir.iterdir():
        if entry.name == MERGED_HF_DIRNAME:
            continue
        if entry.name == GATE_FILENAME and entry != consumed_gate:
            continue  # a stale gate that was not the one registered
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    if not any(export_dir.iterdir()):
        export_dir.rmdir()


def _metadata(name, version, export_dir, gate, training, training_run_dir, quantization):
    dataset_version = (gate or {}).get("dataset_version") or (training or {}).get("dataset_version") or None
    metadata = {
        "name": name,
        "version": version,
        "created_at": _utc_now(),
        "dataset_version": dataset_version,
        "base_model": (training or {}).get("base_model"),
    }
    quantization = quantization or (gate or {}).get("quantization") or (training or {}).get("quantization")
    if quantization:
        metadata["quantization"] = quantization
    # Absolute, so the record still says where the export came from after the shell moves on.
    metadata["source"] = str(Path(export_dir).absolute())
    metadata["gate"] = {"passed": gate.get("passed") is True, "path": GATE_FILENAME} if gate is not None else None
    metadata["training_run"] = _training_summary(training, training_run_dir) if training is not None else None
    return metadata


def _training_summary(training, training_run_dir):
    """The run's path plus what training_metadata.json says, without the bulky parts."""
    summary = {"path": str(Path(training_run_dir).absolute())}
    for key, value in training.items():
        if key != "path" and len(json.dumps(value)) <= _LARGE_TRAINING_FIELD_BYTES:
            summary[key] = value
    return summary


# --- gate and promote ------------------------------------------------------


def read_gate(registry_dir, name, version):
    """The stored gate.json of a version, or None when it has none (or it is unreadable)."""
    path = _version_dir(registry_dir, name, version) / GATE_FILENAME
    gate = _read_json(path)
    if gate is None and path.exists():
        logger.warning("Ignoring unreadable gate file %s", path)
    return gate


def gate_summary(gate):
    """Lines describing a gate verdict: pass/fail, dataset version and each check. No transcript text."""
    if gate is None:
        return ["Gate: none (no readable gate.json)"]
    lines = [f"Gate: {'PASSED' if gate.get('passed') is True else 'NOT PASSED'}"]
    if gate.get("dataset_version"):
        lines.append(f"  dataset_version: {gate['dataset_version']}")
    marks = {True: "PASS", False: "FAIL", None: "SKIP"}  # a skipped check has passed = null
    for check in gate.get("checks") or []:
        if isinstance(check, dict) and "name" in check:
            detail = check.get("detail")
            suffix = f": {detail[:200]}" if isinstance(detail, str) else ""
            lines.append(f"  [{marks.get(check.get('passed'), 'FAIL')}] {check['name']}{suffix}")
    return lines


def promote(registry_dir, name, version, *, force=False, promoted_by="cli"):
    """Make `version` the current one of `name`; returns the new current.json content.

    Refused (GateRefusedError) unless the version's gate exists and passed;
    `force` overrides that and is recorded as `forced: true`, so an
    unevaluated model in service is always visible in the registry.
    """
    version_dir = _version_dir(registry_dir, name, version)
    if not (version_dir / CT2_DIRNAME).is_dir():
        raise RegistryError(f"{name}@{version} is not registered")
    gate = read_gate(registry_dir, name, version)
    if gate is None:
        if not force:
            raise GateRefusedError(f"{name}@{version} has no readable gate.json; evaluate it first or use --force")
    elif gate.get("passed") is not True and not force:
        raise GateRefusedError(f"{name}@{version} did not pass its gate; fix the model or use --force")
    current = {
        "version": version,
        "promoted_at": _utc_now(),
        "promoted_by": promoted_by,
        "forced": bool(force and (gate is None or gate.get("passed") is not True)),
    }
    _write_json_atomic(version_dir.parent / CURRENT_FILENAME, current)
    return current


def current_version(registry_dir, name):
    """The promoted version of `name`, or None when nothing was promoted yet."""
    current = _read_json(_model_dir(registry_dir, name) / CURRENT_FILENAME)
    version = (current or {}).get("version")
    return version if isinstance(version, str) else None


# --- listing and resolving -------------------------------------------------


def _versions(model_dir):
    """Registered version names, oldest first."""
    found = []
    for path in model_dir.iterdir():
        if path.is_dir() and _NAME.fullmatch(path.name) and (path / CT2_DIRNAME).is_dir():
            created_at = (_read_json(path / METADATA_FILENAME) or {}).get("created_at") or ""
            found.append((created_at, path.name))
    return [version for _, version in sorted(found)]


def list_models(registry_dir):
    """[{name, current_version, versions}] for every model in the registry (empty when there is none)."""
    root = Path(registry_dir)
    if not root.is_dir():
        return []
    models = []
    for path in sorted(p for p in root.iterdir() if p.is_dir() and _NAME.fullmatch(p.name)):
        versions = _versions(path)
        if versions:
            models.append({"name": path.name, "current_version": current_version(root, path.name), "versions": versions})
    return models


def show(registry_dir, name, version=None):
    """Details of one model: the current pointer, all versions, and one version's metadata and gate.

    `version` defaults to the current one; with none promoted yet the version
    fields are None and only the version list is filled in.
    """
    model_dir = _model_dir(registry_dir, name)
    versions = _versions(model_dir) if model_dir.is_dir() else []
    if not versions:
        raise RegistryError(f"No model named {name!r} in the registry")
    current = _read_json(model_dir / CURRENT_FILENAME)
    chosen = version or current_version(registry_dir, name)
    result = {"name": name, "current": current, "versions": versions, "version": chosen, "metadata": None, "gate": None}
    if chosen is not None:
        if chosen not in versions:
            raise RegistryError(f"{name} has no version {chosen!r}")
        result["metadata"] = _read_json(model_dir / chosen / METADATA_FILENAME)
        result["gate"] = read_gate(registry_dir, name, chosen)
    return result


def resolve(spec, registry_dir):
    """(path, label) for a model spec.

    `registry:<name>` is the current version's ct2 directory and
    `registry:<name>@<version>` a specific one; the label is always the
    resolved `registry:<name>@<version>`, which is what the service reports
    instead of a filesystem path. Any other spec (a faster-whisper alias, a
    path) is returned unchanged as both path and label.
    """
    if not spec.startswith(REGISTRY_PREFIX):
        return spec, spec
    name, _, version = spec[len(REGISTRY_PREFIX):].partition("@")
    _check_name("model name", name)
    if "@" in spec[len(REGISTRY_PREFIX):] and not version:
        raise RegistryError(f"Invalid model spec {spec!r}: empty version after '@'")
    if not version:
        version = current_version(registry_dir, name)
        if version is None:
            raise RegistryError(f"{name} has no promoted version; promote one or name a version ({REGISTRY_PREFIX}{name}@<version>)")
    ct2 = _version_dir(registry_dir, name, version) / CT2_DIRNAME
    if not ct2.is_dir():
        raise RegistryError(f"{name}@{version} is not registered")
    return str(ct2), f"{REGISTRY_PREFIX}{name}@{version}"
