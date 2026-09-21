# Training pipeline

How a Whisper model is fine-tuned on recordings from crowd-source-voice (csv) and then served by the auditor-stt service. This document describes what the code does. It does not describe a run on real data, and it does not say the pipeline may be used on real voice recordings: read [data-protection.md](data-protection.md) first, in particular "Open items that block real data". The csv side of the interface is in [crowd-source-voice-contract.md](crowd-source-voice-contract.md).

## 1. Data flow

```
crowd-source-voice (csv)                   auditor-stt training system (on-premise)
------------------------                   ------------------------------------------------------------

GET /api/export?format=json  --+
GET <audio_url> (per new row) --+--> dataset sync --> <data-dir>/ledger.sqlite      rows + tombstones
                                     (ffmpeg: 16 kHz mono)   <data-dir>/audio/<sha256>.wav   normalised audio store
                                                  |
                                          dataset build --> <data-dir>/datasets/<version>/
                                                  |            HF DatasetDict (train/dev/test, audio embedded)
                                                  |            manifest.json, build_metadata.json
                                                  |
                                               train ------> <run dir>   LoRA adapter or full model,
                                                  |                       checkpoints, training_metadata.json
                                                  |
                                              export ------> <export dir>   CTranslate2 model (+ _merged_hf for LoRA)
                                                  |
                                                eval ------> <export dir>/gate.json
                                                  |            baseline comparison, normalised WER,
                                                  |            optional long-form regression check
                                                  |
                                     models register ------> <registry>/<name>/<version>/{ct2, metadata.json, gate.json}
                                     models promote  ------> <registry>/<name>/current.json    (refuses a failed or missing gate unless --force)
                                                  |
                     service: AUDITOR_STT_MODEL=registry:<name>[@<version>]     (at startup)
                              POST /model {"model": "registry:<name>"}          (hot swap, needs AUDITOR_STT_API_KEY)
```

The training system reads only csv-origin data. The service (`/inference`, `/v1/audio/transcriptions`, `/v1/jobs`) has no code path to the ledger or the dataset directories, and the training code never reads the job store. Details in [data-protection.md](data-protection.md).

## 2. Where each command runs

| Commands | Needs torch | Where |
|---|---|---|
| `dataset sync`, `dataset build`, `dataset lineage`, `dataset purge`, `dataset prune`, `dataset pull`, `eval` (default `--backend ct2`), `models ...` | no | the normal venv (`pip install -e ".[dev]"`; `dataset build` and `eval` need the `datasets` package, which the `dataset` extra also provides), or the training container |
| `train`, `export`, `eval --backend hf` | yes (`".[training]"`) | the training container in `docker/liturgos-auditor-train` |

`dataset sync` and the long-form check of `eval` need `ffmpeg` on PATH (the training images install it).

On the maintainer's Windows machine, Smart App Control blocks PyTorch's native DLLs (as stated in the training image README; not reproduced while writing this), so `train` and `export` run in the container. The examples below are written as `auditor-stt <command>`. In the container, everything after the image name is the argument list; `docker run` forms, volumes and PowerShell variants are in [docker/liturgos-auditor-train/README.md](../docker/liturgos-auditor-train/README.md). `python -m auditor_stt.cli <command>` is equivalent to `auditor-stt <command>`.

Paths are relative to the current directory. The data directory defaults to `./data` (env `AUDITOR_STT_TRAIN_DATA_DIR`, flag `--data-dir`); in the container it is `/data`. The repository's `.gitignore` covers `/data/` and `/models/` at the repository root only. Run the commands from the repository root, or pass absolute paths outside the repository: `./data` created under `python-packages/liturgos-auditor-stt/` is not git-ignored.

### Default layout

| What | Path |
|---|---|
| Ledger | `<data-dir>/ledger.sqlite` (SQLite in WAL mode, so `-wal` and `-shm` files appear next to it) |
| Normalised audio | `<data-dir>/audio/<sha256>.wav` |
| Dataset versions | `<data-dir>/datasets/<version>/` |
| Training run | whatever `train --out` says; this document uses `models/runs/<run>` |
| CTranslate2 export | whatever `export --out` says; this document uses `models/exports/<name>` |
| Registry | `AUDITOR_STT_REGISTRY_DIR` or `./models/registry` (the compose file sets `/models/registry`) |

## 3. Stages

### 3.1 csv prerequisites

`dataset sync` needs a csv that exports `recording_id`, `audio_url` and a pseudonymous `speaker_id`, and a token. Without `speaker_id` every row is speaker-less, everything is train-only, and `dataset build` refuses to build (it cannot fill dev and test). The fields, the two csv environment variables and the legacy behaviour are in [crowd-source-voice-contract.md](crowd-source-voice-contract.md).

### 3.2 `dataset sync`: csv into the ledger

```
export CSV_ADMIN_TOKEN=...        # csv EXPORT_API_TOKEN, or an admin JWT; never on the command line
auditor-stt dataset sync --base-url https://csv.example.org --corpus-id 1
```

Flags: `--base-url` and `--corpus-id` (required), `--token-env` (name of the environment variable that holds the bearer token, default `CSV_ADMIN_TOKEN`), `--data-dir`, `--speaker-salt-env` (default `AUDITOR_STT_SPEAKER_SALT`), `--allow-mass-removal`. Output is one line, for example `Corpus 1: added 120, updated 0, removed 0, unchanged 0` with `; skipped N (reason=count, ...)` appended when something was skipped. Exit code 0 on success, 1 on any error.

What it does:

- Fetches the validated export (`/api/export?corpus_id=N&format=json`; `include_all` is never sent). Only `text` corpora are accepted.
- New recording: downloads the audio, converts it with ffmpeg to 16 kHz mono PCM16 WAV, checks the length of the converted audio (0.5 to 30 s; csv's own `duration` is client-supplied and is not trusted), stores it as `audio/<sha256>.wav` (identical audio shares one file) and inserts a ledger row.
- Known recording with changed text, quality score or speaker id: the row is updated in place, no download. Changed audio path or a missing audio file: downloaded again.
- Tombstoning: csv hard-deletes and does not say so, so a recording that is no longer in the export is treated as deleted. Its audio file is deleted first (unless another active row has the same audio hash), then the row is tombstoned: text, text hash, audio hash and path, speaker id, duration, score are erased; `recording_id`, `corpus_id`, `first_seen`, `last_seen` and `removed_at` stay. The ledger sets `PRAGMA secure_delete=ON`. A tombstoned recording that reappears in the export is downloaded again.
- Mass-removal guard: before changing anything, the sync refuses when the export lacks more than 5 of the corpus's active ledger rows and also more than half of them (the message says so and names `--allow-mass-removal`). An empty export is treated as an empty listing and falls under the same guard. Use the flag only when the deletions are genuine.
- Legacy csv (rows without `recording_id`): rows are paired with `/api/export/manifest` by positional file name, and the pairing is verified (same count, same normalised text per row). On a mismatch it refetches once and then aborts with "Export and manifest ... still disagree" instead of pairing one recording's text with another's audio.
- Failures are per recording: a failed download or unreadable audio is counted in the `skipped` list and neither aborts the run nor tombstones anything. Skipped recordings have no ledger row, so the next sync tries them again.
- Skip reasons: `download_failed`, `audio_undecodable`, `duration_out_of_range`, `empty_text`, `missing_audio_path`, `raw_speaker_id_dropped`, `unusable_speaker_id_dropped`.
- Speaker ids: the export's pseudonymous `speaker_id` is stored as given. A raw numeric id is hashed with the salt from `AUDITOR_STT_SPEAKER_SALT` or, without a salt, dropped; email-like values are dropped. Raw csv user ids are never stored. See [crowd-source-voice-contract.md](crowd-source-voice-contract.md).
- Idempotent and resumable; each recording is committed on its own. Run one sync at a time per data directory.
- Logs carry recording ids and counts, not transcripts or audio.

### 3.3 `dataset build`: versioned dataset

```
auditor-stt dataset build --corpus-id 1
# Dataset <version>: train N, dev N, test N (N train rows dropped for transcript overlap) -> ./data/datasets/<version>
```

Flags: `--data-dir`, `--corpus-id` (only that corpus's rows), `--out` (parent of the `<version>` directory; default `<data-dir>/datasets`), `--seed` (default 42), `--force`, and `--snapshot DIR` for the legacy path (section 3.9).

Behaviours worth knowing:

- Stable hash split by speaker. `bucket = int(sha256("<salt>:<speaker_id>").hexdigest()[:8], 16) % 10000` with `salt = "auditor-stt-split-v1:<seed>"`. With the fixed ratios 90/5/5, dev is buckets 0 to 499, test is 500 to 999, train is the rest. The split of a speaker depends on nothing but the speaker id, the seed and the ratios, so a speaker never moves between splits as data grows, and test data of one version does not become training data of a later one. Keep `--seed`: changing it reshuffles everyone. The CLI has no flag for the ratios.
- Rows without a speaker id are train-only and are counted in `build_metadata.json` (`train_rows_without_speaker`). They never affect anyone else's split. There is no fallback to a random utterance split any more.
- Transcript-leakage guard. csv prompts are short sentences read by many speakers, so a speaker-disjoint split still shares transcripts between train and test. Every train row whose normalised text also occurs in dev or test is dropped (count: `train_rows_dropped_for_transcript_overlap`). If a small set of prompts is read by every speaker, this can remove most of the training set; the printed count shows it.
- Versioning and idempotence. `<version>` is the first 16 hex characters of a sha256 over the sorted `(recording_id, text_hash, audio_sha256, speaker_id)` tuples plus the split ratios and salt. The same inputs give the same version, and rebuilding an existing version prints "already built; nothing to do" unless `--force`. New data gives a new version directory; older versions are untouched. A directory without `build_metadata.json` is an interrupted build.
- Contents: a HF `DatasetDict` with columns `audio` (16 kHz, the bytes are embedded in the dataset files, not only referenced), `text`, `duration`, `speaker_id`, `recording_id`, `quality_score`; `manifest.json` (per row: `recording_id`, `speaker_id`, `split`, `audio_sha256`, `text_hash`, `duration`; no text); `build_metadata.json` (`dataset_version`, `manifest_sha256`, `created_at`, `source`, `seed`, `split_ratios`, `split_salt`, `split_sizes`, `speaker_counts`, `train_rows_without_speaker`, `train_rows_dropped_for_transcript_overlap`, `speaker_disjoint`).
- Label text goes through `normalize_text`: NFC, literal `\n` and line breaks become spaces, whitespace collapsed. Case and Finnish orthography are kept.
- Errors that stop a build: no ledger ("run `auditor-stt dataset sync` first"); an active row whose audio file is missing (run sync again); a split that would have 0 rows ("Split(s) ['dev'] would have 0 rows ..."). The last one means too few identified speakers: with 5% dev and 5% test, a corpus of only a few speakers usually has nobody in dev or test. The message suggests wider ratios, which the CLI cannot set; the realistic remedy is more speakers.
- Limitation: when a csv account is anonymised (`user_id` set to NULL in csv), the recordings arrive speaker-less on the next sync and become train rows. What was test data in an earlier version can then be training data in a later one, and the test set shrinks, so metrics of the two versions are not comparable for that speaker's share.

### 3.4 `train`

Preset `large-v3-turbo` is the default base (`openai/whisper-large-v3-turbo`); `--preset small` selects `openai/whisper-small`; `--model` (an id or a path) overrides the preset.

```
# LoRA with gradient checkpointing (the option set that can fit an 8 GB card, see section 4)
auditor-stt train --dataset data/datasets/<version> --out models/runs/run-001 \
  --peft lora --gradient-checkpointing --fp16 --batch-size 2 --gradient-accumulation-steps 8

# Small preset, full fine-tune
auditor-stt train --dataset data/datasets/<version> --out models/runs/run-small --preset small --fp16

# Full fine-tune of the default preset (needs a large GPU, see section 4)
auditor-stt train --dataset data/datasets/<version> --out models/runs/run-full --fp16 --gradient-checkpointing --batch-size 1 --gradient-accumulation-steps 16
```

Flags and defaults: `--language fi`, `--task transcribe`, `--learning-rate 1e-5`, `--epochs 3.0`, `--batch-size 4`, `--gradient-accumulation-steps 1`, `--seed 42`, `--fp16`, `--bf16` (the preflight advice names Ampere or newer for bf16), `--peft none|lora` (default `none`), `--lora-r 32`, `--lora-alpha 64`, `--lora-dropout 0.05`, `--lora-target-modules q_proj v_proj`, `--gradient-checkpointing`, `--skip-preflight`. The legacy `scripts/train-whisper.py` takes `--output` instead of `--out`.

Behaviours:

- Preflight: before any weights are downloaded, the estimated GPU memory is compared with the free memory CUDA reports; if it does not fit, `train` stops with the numbers and the options that would help (`--peft lora`, `--preset small`, `--gradient-checkpointing`, `--fp16`/`--bf16`, a smaller `--batch-size`) and mentions `--skip-preflight`. Without a CUDA GPU it only warns that CPU training is extremely slow. The estimates are rough (section 4).
- The trainer sees the train and dev splits only. Dev loss is computed every epoch and the best epoch by dev loss is kept (`load_best_model_at_end`); `save_total_limit=2` keeps at most two `checkpoint-N` directories. The test split is not used in training.
- Labels are the normalised text. The training code adds no timestamp tokens to the labels (see section 5).
- `report_to="none"`: nothing is sent to an external tracker.
- A LoRA run directory holds the adapter (`adapter_config.json`, `adapter_model.safetensors`), the processor files and `generation_config.json`; a full run holds the model. Either way the run directory itself (not a `checkpoint-N`) is what `export` takes.
- `training_metadata.json` records: `base_model`, `base_model_revision`, `preset`, `language`, `task`, `dataset` (absolute path), `dataset_version`, `dataset_manifest_sha256`, `epochs`, `learning_rate`, `seed`, `train_examples`, `dev_examples`, `test_examples`, `train_metrics`, `git_commit`, `package_versions` (torch, transformers, peft, datasets, accelerate), `config`, `peft`. It holds no speaker ids and no text. `git_commit` is `null` when no git checkout is found next to the code, which is the case in the training images (they copy the package without `.git`).
- The dataset directory is written to during training: the feature-preparation step caches its output (log-mel features and label tokens of every utterance) as `cache-*.arrow` files inside the dataset's split directories. That is a large, derived copy of the voice data (a 128-bin, 30 s log-mel array is about 1.5 MB per utterance by arithmetic; not measured). It goes when the dataset version is superseded by `dataset purge` or deleted by `dataset prune`.

The default learning rate is 1e-5 for both modes; the code does not change it for LoRA. Whether that is a good value for adapters on real data has not been tested.

### 3.5 `export`: to CTranslate2

```
auditor-stt export --model models/runs/run-001 --out models/exports/run-001-ct2 --quantization float16 --merge-lora
```

- `--quantization` is `float16` (default), `int8_float16` or `int8`.
- A LoRA run directory is refused without `--merge-lora` ("is a LoRA adapter, not a full model"). With it, the adapter is merged into its base model (loaded from `base_model_name_or_path` in `adapter_config.json`, so the base must be in the HF cache or reachable), the merged fp32 checkpoint is converted, and the merged checkpoint is left in `<out>/_merged_hf`. For a full run `--merge-lora` has no effect.
- `--out` must not exist. `ct2-transformers-converter` must be on PATH; it is installed together with `ctranslate2`, a dependency of faster-whisper.
- `_merged_hf` is a full fp32 copy of the model (about 3 GB for the turbo preset: 809 M parameters times 4 bytes). It is not needed for serving, and `models register` never copies it. It is what `eval --backend hf` needs.

### 3.6 `eval`: the gate

```
auditor-stt eval --model models/exports/run-001-ct2 --dataset data/datasets/<version> \
  --longform-audio /private/sermon.wav --longform-reference /private/sermon.txt --require-longform
```

Exit code 0: gate passed. 3: gate failed (a valid result; `gate.json` is still written). 1: could not evaluate.

Flags: `--model` and `--dataset` (required), `--backend ct2|hf` (default `ct2`), `--split` (default `test`), `--baseline` (default `large-v3-turbo`), `--language fi`, `--device auto|cpu|cuda`, `--compute-type` (default float16 on cuda, int8 on cpu), `--longform-audio`, `--longform-reference`, `--longform-tolerance` (default 0.02), `--min-improvement` (default 0.0), `--require-longform`, `--out` (default `gate.json` inside `--model`), `--dump-predictions PATH`.

What it does:

- Scores the candidate and the zero-shot baseline on the same split. By default both go through faster-whisper/CTranslate2 via the service's `ModelHost`, which is the path that ships. `--backend hf` scores a merged Hugging Face checkpoint (for example `<export>/_merged_hf`) but cannot do the long-form check, and a LoRA adapter directory is refused. The baseline is always scored through CT2, and it is an alias or a CT2 directory: it is the zero-shot model, not the model currently in service, unless you pass the current export as `--baseline`. The baseline alias is downloaded on first use unless it is already cached; the download directory is `AUDITOR_STT_MODEL_DIR` when set (the service's cache), otherwise the default Hugging Face cache.
- Metrics per model: normalised WER and CER (lowercase, punctuation and symbols to spaces, digits kept), raw WER and CER, and per-speaker normalised WER as a spread (min, median, p90, max; speakers with under 30 s of audio are flagged and also reported separately). Relative improvement is `(baseline - candidate) / baseline`.
- Check `beats_baseline_on_test`: the candidate's normalised WER must be below `baseline * (1 - min_improvement)`.
- Check `longform_no_regression`, described in section 5. Without `--longform-audio` and `--longform-reference` this check is skipped, and a skipped check does not fail the gate. Use `--require-longform` to make its absence a failure. `--longform-audio` and `--longform-reference` must be given together.
- `passed` is true when every check that was not skipped passed.
- `gate.json` holds metrics and verdicts only: no transcript text, no speaker ids, no recording ids. `--dump-predictions PATH` writes every reference and hypothesis (with recording and speaker ids) as JSON lines; those are people's words, so the option is off by default and the file needs the same care as the dataset.
- Console output and logs carry numbers only.

### 3.7 `models`: registry

```
auditor-stt models register --name whisper-fi --version run-001 \
  --export models/exports/run-001-ct2 --training-run models/runs/run-001
auditor-stt models promote --name whisper-fi --version run-001
auditor-stt models list
auditor-stt models show --name whisper-fi
```

`register` also takes `--gate` (default: `gate.json` inside `--export`), `--quantization` (recorded in the metadata), `--move` (remove the copied files from `--export` once registered; `_merged_hf` is kept), and `--registry-dir` (all four subcommands; default `AUDITOR_STT_REGISTRY_DIR` or `./models/registry`).

- Layout: `<registry>/<name>/<version>/ct2/`, `metadata.json`, `gate.json`, and `<registry>/<name>/current.json`. No symlinks (Windows and Docker volumes). Names and versions match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. A version is never overwritten; the CT2 export is copied without `_merged_hf` and `gate.json`.
- `metadata.json`: `name`, `version`, `created_at`, `dataset_version`, `base_model`, `quantization` (if known), `source` (the export path), `gate` (`passed`, `path`; null when there is no gate), `training_run` (the run's `training_metadata.json` without bulky fields).
- `promote` prints the gate summary and exits 0 (done), 3 (refused by the gate) or 1 (error). It refuses a version whose gate is missing, unreadable or not passed, unless `--force`; a forced promotion is recorded as `"forced": true` in `current.json`. `current.json` also holds `promoted_at` and `promoted_by`; `promoted_by` is the literal `cli`, not a person, so who promoted needs another record.
- There is no command to delete or un-register a version; remove the directory by hand.

### 3.8 Serving the model

At startup, `AUDITOR_STT_MODEL=registry:whisper-fi` (the promoted version) or `registry:whisper-fi@run-001` (a specific one). The service then reports `model_id` as `registry:whisper-fi@run-001` on `/health`, not a filesystem path. A registry spec that cannot be resolved does not stop the service: `/health` stays at 503 until a model is switched in.

Without a restart:

```
curl -X POST http://localhost:8090/model \
  -H "Authorization: Bearer $AUDITOR_STT_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "registry:whisper-fi"}'
curl -H "Authorization: Bearer $AUDITOR_STT_API_KEY" http://localhost:8090/model     # current model + registry contents
```

- `POST /model` is refused with 403 unless `AUDITOR_STT_API_KEY` is set. It accepts `registry:<name>[@<version>]` and the aliases listed in `AUDITOR_STT_ALLOWED_MODELS` (comma separated); paths are never accepted (422).
- The new model is loaded while the old one keeps serving, then swapped in. A failed load leaves the old model serving (500). A second switch during a switch gets 409. Requests already running finish on the old model. Old and new model are resident together during the swap, so the device needs room for both.
- A batch job that spans a switch continues on the new model chunk by chunk; the result lists every model used (`models`).
- The registry directory must be one the service can see: in `docker-compose.yml` it is `/models/registry` on the `auditor-stt-models` volume.

### 3.9 Legacy snapshots

`auditor-stt dataset pull --base-url ... --corpus-id N --out DIR` still writes a snapshot (`recordings.json`, `audio/` with the original audio bytes, `snapshot.json`) and `dataset build --snapshot DIR --out DIR2` builds from it (`--out` is required there and is the output directory itself). Snapshots have no ledger, no tombstoning, positional file names as ids, and take `speaker_id` as the export gave it. Datasets built this way and snapshots are not seen by `dataset lineage`, `purge` and `prune`. Prefer `dataset sync`.

## 4. Hardware reality

The dev machine has an 8 GB RTX 2060 SUPER. The GPU that will be used for real training is deliberately undecided; the pipeline is hardware-agnostic and offers options instead of choosing.

A full fine-tune of `large-v3-turbo` (809 M parameters) needs fp32 weights, gradients and two Adam states (16 bytes per parameter, about 12 GiB) before any activations. The preflight refuses it on the dev card, whatever batch size. Estimates from `auditor_stt/training/preflight.py` (rough by design, in GB; the constants are in that file; none was measured on a card):

| Setup | Estimate |
|---|---|
| turbo, full, fp32, batch 4 (the defaults of `train`) | 37.2 |
| turbo, full, fp16, batch 1 | 17.6 |
| turbo, full, fp16 + checkpointing, batch 1 | 15.0 |
| turbo, LoRA, fp32, batch 4, no checkpointing | 28.4 |
| turbo, LoRA + checkpointing, fp32, batch 4 | 7.9 |
| turbo, LoRA + checkpointing + fp16, batch 4 | 7.6 |
| turbo, LoRA + checkpointing + fp16, batch 2 | 6.7 |
| turbo, LoRA + checkpointing + fp16, batch 1 | 6.2 |
| small, full, fp16 + checkpointing, batch 4 | 5.6 |
| small, LoRA + checkpointing + fp16, batch 4 | 3.0 |

The preflight compares with the free memory, and a card that also drives a display has less than 8 GB free, so the LoRA rows for turbo are borderline on the dev card. The options that are realistic there are LoRA with gradient checkpointing and fp16 at a small batch size (raise `--gradient-accumulation-steps` to keep the effective batch), or `--preset small`. 8-bit optimisers are not used (bitsandbytes is not relied on under Windows). RTX 20-series cards have no native bf16, so use `--fp16`.

Training on CPU works but the code itself warns that it is extremely slow (days for the turbo preset). It is suitable for a mechanics test, not for a real run: `AUDITOR_STT_RUN_SLOW=1 pytest tests/test_training_smoke.py` runs train and export on `openai/whisper-tiny` with synthetic audio, with and without LoRA. It checks that the pieces fit together, not that a model learns.

Running the CUDA image needs the NVIDIA Container Toolkit (per the Dockerfile header). Whether the dev card is reachable from a container on the maintainer's machine has not been verified in this repository.

## 5. Domain mismatch and the long-form check

The csv data are short sentences read aloud (3 to 15 words according to the project plan; csv's README gives 0.5 to 120 s for a recording, checked in the browser, and the sync keeps 0.5 to 30 s), recorded in browsers. The target is long, spontaneous sermon speech in a church acoustic, with liturgical vocabulary. Two risks follow:

- A model tuned on read sentences may not improve long-form speech at all, or may get worse (forgetting, register mismatch).
- The training labels carry no timestamp tokens (the code adds none). A fine-tuned model may therefore lose segment timing or long-form decoding behaviour even if it wins on the test split. Word timestamps come from the alignment heads and are expected to survive; that expectation is untested here.

The test split is more read speech from other speakers, so a good number there says nothing about sermons. That is why the gate has a long-form regression check, `longform_no_regression`:

- It runs the candidate and the baseline (one after the other, so one model is in memory at a time) through the same batch pipeline the service uses for jobs: ffmpeg to 16 kHz mono, 60 s chunks snapped to pauses, the tail of the previous chunk carried as prompt.
- It compares the normalised WER of each against a hand-corrected reference text of the same audio. Regression if the candidate's WER exceeds the baseline's by more than `--longform-tolerance` (default 0.02, absolute).
- It also checks that timestamps look like a transcript's: more out-of-order or overlapping segments than the baseline, a share of segments of 25 s or longer that more than doubles (and grows by at least 5 points), or the last segment ending more than 10 points earlier relative to the audio length, each count as a regression. This is a structural sanity check; it does not compare timestamps against a reference.
- If the baseline finds no speech in the audio, the check aborts (nothing to compare against).
- Nothing textual is returned or written; the temporary WAV lives in a temporary directory that is removed afterwards.

The check needs a private, hand-corrected reference: a UTF-8 text file with the correct transcript of the audio you pass (a byte-order mark is accepted; 30 to 60 minutes of a real sermon is what the plan calls for). The audio and the reference are not in git and must stay out of it. The repository's `.gitignore` ignores `*.mp4`, `/data/` and `/models/`, not `.wav` or `.txt`, so keep them outside the repository, for example in a private directory mounted read-only into the container. This audio is speech of real people who are not csv volunteers; see [data-protection.md](data-protection.md) for what that implies. It is used for evaluation only; the code never adds it to a dataset. Until a reference exists, the gate can still pass on the test split alone, and that pass is weaker evidence than it looks.

## 6. Retention and erasure

Ledger rows, audio, datasets, runs, exports and registry versions do not expire by themselves; the only automatic clean-up in the training system is `save_total_limit=2` for checkpoints during a run. Retention, withdrawal and erasure are in [data-protection.md](data-protection.md): `dataset lineage --speaker`, `dataset purge --speaker`, `dataset prune`.

## 7. What changed from the earlier version of this document

- Pulling a snapshot is replaced by `dataset sync` into a ledger. `dataset pull` is kept only for snapshot compatibility.
- The old statement that builds fall back to a deterministic utterance-level split when a speaker id is missing is gone from the code. Speaker-less rows are train-only, and a corpus without identified speakers cannot fill dev and test, so the build stops.
- `scripts/evaluate-whisper.py` (now `python -m auditor_stt.training.evaluate`) is the old single-checkpoint scorer: it scores a Hugging Face checkpoint and writes `test_metrics.json` with `wer` (raw, as before) plus normalised numbers. The gate is `auditor-stt eval`, described above, with baseline, normalised WER, long-form check and `gate.json`; `models promote` reads that file.
- The old training example used a full fine-tune of the turbo model with `--fp16 --batch-size 4`. The preflight now refuses that on small GPUs.
- The test split is still not used by the trainer. The dev split is (loss each epoch, best-epoch selection).

## 8. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `error: Env var CSV_ADMIN_TOKEN is not set` | Export the variable (or the one named by `--token-env`). In `docker run` use `-e CSV_ADMIN_TOKEN` so the host value is passed. |
| sync fails with HTTP 401 or 403 | Wrong or expired token (an admin JWT lives 7 days; the csv export token does not expire). The export token works only on `/api/export`, `/api/export/manifest`, `/api/export/stats`. |
| `Refusing to remove N of M recordings` | The export lost more than half of the corpus. Check `--corpus-id` and the csv side first; add `--allow-mass-removal` only for genuine deletions. |
| `Export and manifest ... still disagree after a refetch` | Legacy csv only: something changed between the two calls. Run again when no validation is in progress, or use a csv that exports `recording_id`. |
| `Corpus N ... is type 'music'` | Only `text` corpora are trained on. |
| `error: ffmpeg was not found on PATH` | Install ffmpeg (the training images have it). |
| Many `skipped` reasons in the sync line | See the reason list in section 3.2. `duration_out_of_range` means the converted audio is outside 0.5 to 30 s; skipped recordings are retried on every sync. |
| SQLite "database is locked" | Another sync or command holds the ledger. One at a time per data directory. |
| `dataset build`: `No ledger at ...` | Run `dataset sync` first (and check `--data-dir`). |
| `dataset build`: `active recordings have no audio file` | The audio store lost files; run `dataset sync` again, it downloads them. |
| `dataset build`: `Split(s) [...] would have 0 rows` | Too few identified speakers for dev and test. Sync more data. Changing `--seed` may move a speaker in, but it also reshuffles the whole split, so do it only for the first build. |
| Most training rows dropped for transcript overlap | Every speaker reads the same few prompts; the leakage guard removes the train rows whose sentence is in dev or test. Use a corpus with more distinct prompts. |
| `train`: preflight refuses the run | Read the options in the message; see section 4. `--skip-preflight` overrides at your own risk (CUDA out of memory). |
| `train`: torch import fails on the Windows host | Expected there (Smart App Control); run in the training container. |
| Disk fills up during or after training | Dataset caches (`cache-*.arrow`) inside the dataset directory, up to two `checkpoint-N` per run (with optimiser state), `_merged_hf` in exports. `dataset prune --runs-dir models/runs --yes` removes checkpoints of finished runs. |
| `training_metadata.json` has `"git_commit": null` | No git checkout in the image. Tag the image and record the tag yourself. |
| `export`: `is a LoRA adapter, not a full model` | Add `--merge-lora`. |
| `export`: `already exists; choose a new output directory` | The converter refuses an existing directory. |
| `export`: `does not name its base model` | `adapter_config.json` lacks `base_model_name_or_path`; the run is not from this trainer. |
| `eval`: `is not a CTranslate2 model directory (no model.bin)` | Run `export` first, or use `--backend hf` on a merged checkpoint. |
| `eval`: `is a LoRA adapter, not a model` (hf backend) | Evaluate `<export>/_merged_hf`, or use the default ct2 backend on the export. |
| `eval`: `The long-form check needs both an audio file and a reference text` | Pass `--longform-audio` and `--longform-reference` together, and use the ct2 backend. |
| `eval`: `baseline model found no speech in the long-form audio` | The audio or its decoding is wrong; the check cannot compare against silence. |
| `eval`: `is not a directory, so there is nowhere to put gate.json` | `--model` is an alias or path that is not a directory; pass `--out`. |
| `eval` exits 3 | The gate failed; the reasons are printed and stored in `gate.json`. This is a result, not an error. |
| `eval` on the CUDA image fails to use the GPU | See the comments in `docker/liturgos-auditor-train/Dockerfile.cuda`. |
| `models promote` exits 3 | The version has no gate or a failed gate. Run `eval`, register again as a new version with its `gate.json`, or use `--force` (recorded as forced). |
| `models register`: `is already registered; versions are never overwritten` | Pick a new `--version`. |
| `POST /model` returns 403 | Set `AUDITOR_STT_API_KEY` on the service. |
| `POST /model` returns 422 | The spec must be `registry:<name>[@<version>]` or listed in `AUDITOR_STT_ALLOWED_MODELS`; `has no promoted version` means promote first or name a version. |
| `POST /model` returns 500 | The new model failed to load; the old one is still serving (`/health`). |
