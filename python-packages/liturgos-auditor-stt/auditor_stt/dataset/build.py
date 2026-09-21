"""`auditor-stt dataset build` — versioned HF datasets with a stable, leakage-guarded split.

Reads the active rows of the sync ledger (or, for compatibility, a `dataset
pull` snapshot) and writes `<out>/<version>/`: a DatasetDict with train/dev/test,
`manifest.json` (which recordings went in, never their text) and
`build_metadata.json`. Three properties keep evaluation trustworthy as the
corpus grows:

* The split is a pure function of the speaker id and a salt (a hash bucket),
  never of what else is in the corpus. Adding speakers or recordings cannot
  move an existing speaker, so v2's test set never leaks into v3's training
  data and metrics stay comparable across versions.
* A recording without a speaker id (csv anonymised recordings) can only be
  train. It never degrades the split of everyone else.
* csv prompts are short sentences read by many speakers, so a speaker-disjoint
  split still shares transcripts between train and test. Train rows whose text
  also occurs in dev or test are dropped.

`dataset_version` hashes the inputs and the split configuration, so identical
inputs give an identical version and rebuilding it is a no-op.

Privacy: logs carry counts and recording ids only, never transcripts or audio.
"""

import hashlib
import json
import locale
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .audio import audio_path
from .ledger import Ledger, ledger_path, text_hash, utc_iso
from .normalize import normalize_text

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_RATIOS = {"train": 0.9, "dev": 0.05, "test": 0.05}
SPLIT_NAMES = ("train", "dev", "test")

DATASETS_DIRNAME = "datasets"
MANIFEST_FILENAME = "manifest.json"
BUILD_METADATA_FILENAME = "build_metadata.json"
SUPERSEDED_FILENAME = "SUPERSEDED.json"
MANIFEST_VERSION = 1

_BUCKETS = 10000
# Names that identify a directory as one of our dataset builds, complete or not.
_OWN_NAMES = (BUILD_METADATA_FILENAME, MANIFEST_FILENAME, SUPERSEDED_FILENAME, "dataset_dict.json", *SPLIT_NAMES)


def datasets_root(data_dir):
    return Path(data_dir) / DATASETS_DIRNAME


def split_salt(seed):
    return f"auditor-stt-split-v1:{seed}"


def speaker_bucket(speaker_id, salt):
    return int(hashlib.sha256(f"{salt}:{speaker_id}".encode("utf-8")).hexdigest()[:8], 16) % _BUCKETS


def _validated_ratios(split_ratios):
    ratios = dict(split_ratios or DEFAULT_SPLIT_RATIOS)
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"split_ratios must name exactly {list(SPLIT_NAMES)}, got {sorted(ratios)}")
    if any(value < 0 for value in ratios.values()):
        raise ValueError(f"split_ratios must not be negative: {ratios}")
    return {name: float(ratios[name]) for name in SPLIT_NAMES}


def speaker_split(speaker_id, salt, split_ratios):
    """The split a speaker belongs to. dev, then test, are carved from the low end of the
    bucket range; train is the rest. Depends on nothing but its arguments."""
    dev_end = round(split_ratios["dev"] * _BUCKETS)
    test_end = dev_end + round(split_ratios["test"] * _BUCKETS)
    bucket = speaker_bucket(speaker_id, salt)
    if bucket < dev_end:
        return "dev"
    return "test" if bucket < test_end else "train"


def compute_split(rows, split_ratios=None, seed=42, *, key="file"):
    """Returns (assignment: {row[key]: split_name}, speaker_disjoint: bool).

    Rows without a `speaker_id` are train. `speaker_disjoint` is true when no
    speaker appears in more than one split; the hash split guarantees it,
    the check keeps the flag honest.
    """
    ratios = _validated_ratios(split_ratios)
    salt = split_salt(seed)
    assignment = {}
    splits_of = {}
    for row in rows:
        speaker = row.get("speaker_id")
        split = speaker_split(speaker, salt, ratios) if speaker else "train"
        assignment[row[key]] = split
        if speaker:
            splits_of.setdefault(speaker, set()).add(split)
    return assignment, all(len(splits) == 1 for splits in splits_of.values())


def build_from_ledger(data_dir, *, out_root=None, corpus_id=None, seed=42, split_ratios=None, force=False, max_drop_fraction=None):
    """Build a dataset from the active ledger rows (optionally one corpus) into `<out_root>/<version>`.

    `out_root` defaults to `<data_dir>/datasets`. Returns the build metadata.
    An already-built version is returned as is, unless `force`. Raises
    FileNotFoundError when there is no ledger or an active row has no audio
    file (a sync fetches it again), ValueError when a split would be empty or
    when max_drop_fraction is exceeded.
    """
    rows, corpus_ids = _ledger_rows(data_dir, corpus_id)
    plan, leakage_metadata = _plan(rows, split_ratios, seed, max_drop_fraction=max_drop_fraction)
    out_dir = Path(out_root) if out_root is not None else datasets_root(data_dir)
    return _materialise(plan, out_dir / plan.version, {"kind": "ledger", "corpus_ids": corpus_ids}, seed, force, leakage_metadata)


def build_dataset(snapshot_dir, out_dir, seed=42, split_ratios=None, force=False, max_drop_fraction=None):
    """Build a dataset from a `dataset pull` snapshot (`recordings.json` + `audio/`) into `out_dir` itself.

    A snapshot has no recording ids or audio hashes, so the file name (e.g.
    `0001.wav`, positional within that snapshot) stands in for the recording id
    and the sha256 of the file's bytes for the audio hash. Snapshots pulled
    before csv exported `speaker_id` have none, so every row is train-only and
    the build fails the empty dev/test guard by design.

    `out_dir` is replaced if it holds a different earlier build (a directory
    that does not look like one is never touched).
    """
    snapshot_dir = Path(snapshot_dir)
    rows = _snapshot_rows(snapshot_dir)
    plan, leakage_metadata = _plan(rows, split_ratios, seed, max_drop_fraction=max_drop_fraction)
    return _materialise(plan, Path(out_dir), _snapshot_source(snapshot_dir), seed, force, leakage_metadata)


@dataclass
class _Plan:
    version: str
    salt: str
    ratios: dict
    rows: list  # rows that go into the dataset, each with its "split"
    dropped_for_overlap: int
    speaker_disjoint: bool


def _read_text(path):
    """Read text the way `dataset pull` wrote it: UTF-8, or the locale encoding for older snapshots."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode(locale.getpreferredencoding(False))


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ledger_rows(data_dir, corpus_id):
    path = ledger_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"No ledger at {path}; run `auditor-stt dataset sync` first")
    with Ledger(path) as ledger:
        active = ledger.active_rows(corpus_id)

    rows, missing = [], []
    for row in active:
        wav = audio_path(data_dir, row["audio_sha256"])
        if not wav.is_file():
            missing.append(row["recording_id"])
            continue
        text = normalize_text(row["text"])
        rows.append({
            "recording_id": row["recording_id"],
            "speaker_id": row["speaker_id"] or None,
            "text": text,
            "text_hash": text_hash(text),
            "audio_sha256": row["audio_sha256"],
            "audio": str(wav),
            "duration": row["duration"],
            "quality_score": row["quality_score"],
        })
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} active recordings have no audio file under {Path(data_dir) / 'audio'} "
            f"(recording ids: {missing[:10]}{'...' if len(missing) > 10 else ''}); "
            "run `auditor-stt dataset sync` to fetch them again"
        )
    return rows, sorted({row["corpus_id"] for row in active})


def _snapshot_rows(snapshot_dir):
    entries = json.loads(_read_text(snapshot_dir / "recordings.json"))
    rows = []
    for entry in entries:
        wav = snapshot_dir / "audio" / entry["file"]
        text = normalize_text(entry["text"])
        rows.append({
            "recording_id": entry["file"],
            "speaker_id": entry.get("speaker_id") or None,
            "text": text,
            "text_hash": text_hash(text),
            "audio_sha256": _sha256_file(wav),
            "audio": str(wav),
            "duration": entry["duration"],
            "quality_score": entry.get("quality_score"),
        })
    return rows


def _snapshot_source(snapshot_dir):
    source = {"kind": "snapshot", "corpus_ids": [], "snapshot_content_hash": None}
    info_path = snapshot_dir / "snapshot.json"
    if info_path.is_file():
        info = json.loads(_read_text(info_path))
        if info.get("corpus_id") is not None:
            source["corpus_ids"] = [info["corpus_id"]]
        source["snapshot_content_hash"] = info.get("content_hash")
    return source


def _dataset_version(rows, ratios, salt):
    """First 16 hex of sha256 over the sorted (id, text hash, audio hash, speaker) tuples and the split config."""
    payload = {
        "v": 1,
        "rows": [[r["recording_id"], r["text_hash"], r["audio_sha256"], r["speaker_id"]] for r in rows],
        "split": {"ratios": ratios, "salt": salt},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _plan(rows, split_ratios, seed, max_drop_fraction=None):
    if not rows:
        raise ValueError("No recordings to build a dataset from")
    ratios = _validated_ratios(split_ratios)
    salt = split_salt(seed)
    rows = sorted(rows, key=lambda r: r["recording_id"])
    version = _dataset_version(rows, ratios, salt)

    assignment, speaker_disjoint = compute_split(rows, ratios, seed, key="recording_id")
    eval_texts = {r["text_hash"] for r in rows if assignment[r["recording_id"]] != "train"}

    # Count train rows before the leakage guard.
    train_rows_before = sum(1 for r in rows if assignment[r["recording_id"]] == "train")

    kept, dropped = [], 0
    for row in rows:
        split = assignment[row["recording_id"]]
        if split == "train" and row["text_hash"] in eval_texts:
            dropped += 1
        else:
            kept.append({**row, "split": split})

    # Compute drop fraction and check against max_drop_fraction.
    drop_fraction = (dropped / train_rows_before) if train_rows_before > 0 else 0
    if max_drop_fraction is not None and drop_fraction > max_drop_fraction:
        raise ValueError(
            f"Dropped {dropped} of {train_rows_before} train rows for transcript overlap ({drop_fraction:.1%} > "
            f"max {max_drop_fraction:.1%}). This usually means the same prompts are read by many speakers; "
            "use a corpus with more distinct prompts or build dev/test from held-out sentences instead."
        )

    sizes = {name: sum(1 for r in kept if r["split"] == name) for name in SPLIT_NAMES}
    empty_splits = [name for name, count in sizes.items() if count == 0]
    if empty_splits:
        # Left unchecked, an empty split reaches `datasets`' save_to_disk() and crashes with an
        # opaque ZeroDivisionError in _estimate_nbytes() for any empty split with an Audio column.
        speakers = {r["speaker_id"] for r in rows if r["speaker_id"]}
        raise ValueError(
            f"Split(s) {empty_splits} would have 0 rows with split_ratios={ratios} "
            f"({len(rows)} rows from {len(speakers)} speakers, {sum(1 for r in rows if not r['speaker_id'])} "
            f"without a speaker id, {dropped} train rows dropped for transcript overlap). Rows without a speaker "
            "id only ever go to train, so dev/test need recordings from more identified speakers: use a larger "
            "snapshot or corpus, or wider dev/test ratios."
        )

    # Warn if a large fraction was dropped.
    if drop_fraction > 0.5:
        logger.warning(
            "Dropped %d of %d train rows (%.1f%%) for transcript overlap. This usually means the same prompts "
            "are read by many speakers; consider building dev/test from held-out sentences instead.",
            dropped, train_rows_before, drop_fraction * 100
        )

    # Compute speakers lost in training set.
    speakers_before = {r["speaker_id"] for r in rows if assignment[r["recording_id"]] == "train" and r["speaker_id"]}
    speakers_after = {r["speaker_id"] for r in kept if r["split"] == "train" and r["speaker_id"]}
    speakers_lost = len(speakers_before - speakers_after)

    # Compute distinct transcripts in training set after guard.
    train_distinct_transcripts = len({r["text_hash"] for r in kept if r["split"] == "train"})

    return _Plan(version, salt, ratios, kept, dropped, speaker_disjoint), {
        "train_rows_before_leakage_guard": train_rows_before,
        "train_dropped_fraction": drop_fraction,
        "train_distinct_transcripts": train_distinct_transcripts,
        "train_speakers_lost": speakers_lost,
    }


def _read_metadata(out_dir):
    path = out_dir / BUILD_METADATA_FILENAME
    if not path.is_file():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return metadata if isinstance(metadata, dict) else None


def _clear_output_dir(out_dir):
    if not out_dir.exists():
        return
    if any(out_dir.iterdir()) and not any((out_dir / name).exists() for name in _OWN_NAMES):
        raise ValueError(f"{out_dir} is not empty and does not look like a built dataset; refusing to overwrite it")
    shutil.rmtree(out_dir)


def _materialise(plan, out_dir, source, seed, force, leakage_metadata=None):
    existing = _read_metadata(out_dir)
    if existing is not None and existing.get("dataset_version") == plan.version and not force:
        logger.info("Dataset version %s is already built; nothing to do", plan.version)
        return existing

    from datasets import Audio, Dataset, DatasetDict, Features, Value

    # build_metadata.json is written last: a directory without it is an interrupted build.
    _clear_output_dir(out_dir)
    out_dir.mkdir(parents=True)

    id_type = "int64" if all(isinstance(r["recording_id"], int) for r in plan.rows) else "string"
    features = Features({
        "audio": Value("string"),
        "text": Value("string"),
        "duration": Value("float64"),
        "speaker_id": Value("string"),
        "recording_id": Value(id_type),
        "quality_score": Value("float64"),
    })
    splits = {}
    for name in SPLIT_NAMES:
        splits[name] = Dataset.from_list(
            [
                {column: r[column] for column in features}
                for r in plan.rows
                if r["split"] == name
            ],
            features=features,
        ).cast_column("audio", Audio(sampling_rate=16000))
    DatasetDict(splits).save_to_disk(str(out_dir))

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version": plan.version,
        "rows": [
            {
                "recording_id": r["recording_id"],
                "speaker_id": r["speaker_id"],
                "split": r["split"],
                "audio_sha256": r["audio_sha256"],
                "text_hash": r["text_hash"],
                "duration": r["duration"],
            }
            for r in plan.rows
        ],
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (out_dir / MANIFEST_FILENAME).write_bytes(manifest_bytes)

    metadata = {
        "dataset_version": plan.version,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "created_at": utc_iso(),
        "source": source,
        "seed": seed,
        "split_ratios": plan.ratios,
        "split_salt": plan.salt,
        "split_sizes": {name: len(splits[name]) for name in SPLIT_NAMES},
        "speaker_counts": {
            name: len({r["speaker_id"] for r in plan.rows if r["split"] == name and r["speaker_id"]})
            for name in SPLIT_NAMES
        },
        # After the leakage guard: rows dropped for transcript overlap are in neither count.
        "train_rows_without_speaker": sum(1 for r in plan.rows if r["split"] == "train" and not r["speaker_id"]),
        "train_rows_dropped_for_transcript_overlap": plan.dropped_for_overlap,
        "speaker_disjoint": plan.speaker_disjoint,
    }

    # Add leakage guard metadata.
    if leakage_metadata:
        metadata.update(leakage_metadata)
        # Add warnings list if drop fraction is high.
        if leakage_metadata.get("train_dropped_fraction", 0) > 0.5:
            metadata["warnings"] = [
                f"Dropped {plan.dropped_for_overlap} train rows ({leakage_metadata['train_dropped_fraction']:.1%}) "
                f"for transcript overlap; {leakage_metadata['train_speakers_lost']} speakers lost all their training rows. "
                f"This usually means the same prompts are read by many speakers; consider building dev/test from "
                f"held-out sentences instead."
            ]

    (out_dir / BUILD_METADATA_FILENAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Built dataset %s: sizes %s, %d train rows dropped for transcript overlap",
        plan.version, metadata["split_sizes"], plan.dropped_for_overlap,
    )
    return metadata
