# Data protection

What personal data the liturgos-auditor training system holds, where it is, and what the code does about it. This document is not legal advice. It describes the behaviour of the code as read on the date of writing. It does not say the system is compliant with the GDPR or with any other rule, and it does not decide the questions that belong to the data controller and the data protection officer (DPO): lawful basis, retention periods, consent wording, roles, whether a data protection impact assessment (DPIA) is needed. Those are listed in "Open items that block real data" and in the retention and roles sections.

The audience is the maintainer, who is also the controller's technical contact, and a DPO reviewing the system.

## 1. Purpose and the two data streams

The system has two separate uses of audio:

| Stream | Source | What is done with it | Kept |
|---|---|---|---|
| Training | Volunteer recordings from crowd-source-voice (csv): a pseudonymous speaker id, the prompt text they read, and the voice sample | Synced into a local ledger, built into datasets, used to fine-tune and evaluate a Finnish speech-to-text model | Until someone deletes it (nothing expires by itself; section 8) |
| Serving | Audio sent by lcyt (live caption chunks) and saarnavideo (whole services or sermons) to the transcription service | Transcribed and returned | Live requests: only for the duration of the request. Batch jobs: the audio is deleted when the job ends; the transcript is kept for a limited time (section 2) |

The purpose as implemented: csv recordings are used to train and evaluate the speech recognition model that the organisation's own transcription service runs. Audio that passes through the service is never training data. Using it for that would need its own lawful basis and an explicit decision, and the code offers no way to do it.

The purpose as told to csv contributors is a different question (see open item 5): the consent gate says the recordings "will be used to create speech recognition and synthesis datasets" and "may be shared publicly for research purposes".

Special-category data: whether recordings of people reading religious texts, or the fact that a person volunteers for a church project, reveal religious belief or affiliation is a question for the DPO (open item 6). The controls below do not depend on the answer (they are the same either way); the answer decides the lawful basis and the safeguards that are required.

## 2. Personal data held, and where

Defaults are shown; every path can be changed by flags or environment variables.

| Item | Location | Contents | Notes |
|---|---|---|---|
| csv source (outside this repository) | csv database and `uploads/` | Emails, password hashes, consent timestamps, recordings, user ids | Not touched by this system except through the export. See [crowd-source-voice-contract.md](crowd-source-voice-contract.md) |
| Ledger | `<data-dir>/ledger.sqlite` (+ `-wal`, `-shm`) | Per recording: `recording_id` (csv's integer id), `corpus_id`, `speaker_id` (pseudonym or NULL), `text` (the prompt as read), `text_hash`, `audio_sha256`, `audio_path` (csv's relative file path), `duration`, `quality_score`, `validated_at` (column exists; sync does not fill it), `status`, `first_seen`, `last_seen`, `removed_at` | Nothing email-like. A tombstoned row keeps only `recording_id`, `corpus_id`, `status` and the three timestamps |
| Normalised audio store | `<data-dir>/audio/<sha256>.wav` | The voice recordings, converted to 16 kHz mono WAV | Named by the hash of the audio; identical audio shares a file |
| Dataset versions | `<data-dir>/datasets/<version>/` | HF dataset files with the audio bytes embedded (a copy of the voice data), text, `speaker_id`, `recording_id`, `quality_score`; `manifest.json` (per row: recording id, speaker id, split, audio hash, text hash, duration; no text); `build_metadata.json` (counts, seed, ratios) | After `train`, also `cache-*.arrow` files with the log-mel features and label tokens of every utterance, inside the dataset directory |
| Legacy snapshots | wherever `dataset pull --out` pointed | `recordings.json` (text, speaker id as the export gave it), `audio/` with the original audio bytes, `snapshot.json` | Only if `dataset pull` was used. Not managed by the ledger |
| Training runs | wherever `train --out` pointed | Model weights or LoRA adapter, at most two `checkpoint-N` directories per run (with optimiser state), `training_metadata.json` | Weights can memorise training utterances (section 5). The metadata names the dataset version and manifest hash, counts and configuration; no speaker ids, no text |
| Exports | wherever `export --out` pointed | CTranslate2 model, `gate.json` (metrics and verdicts only), and for LoRA runs `_merged_hf/` (a full fp32 checkpoint) | |
| Registry | `AUDITOR_STT_REGISTRY_DIR` (`./models/registry`; `/models/registry` in compose) | Per version: the CT2 model, `metadata.json` (dataset version, base model, a summary of the training metadata, gate verdict), `gate.json`; `current.json` | No speaker ids, no text. `promoted_by` is the literal `cli`, not a person |
| Evaluation output (optional) | path given to `eval --dump-predictions` | Every reference and hypothesis text with recording and speaker ids | Off by default; treat like the dataset |
| Long-form evaluation audio (optional) | wherever `eval --longform-audio` and `--longform-reference` point | A recording of a real sermon or service and its hand-corrected transcript | Not csv data; see section 4 |
| HF cache | `HF_HOME` (`/hf` in the training images) | Public base-model weights | Should hold no personal data; the training code does not put any there |
| Service job store | `<AUDITOR_STT_DATA_DIR>/jobs/<id>/` | `manifest.json` (parameters, including any `prompt` and `client_ref` the caller sent), `chunks/*.json` (transcript text and timestamps), while running also `source/` (an uploaded input) and `audio.wav` (the normalised audio) | The upload copy and the WAV are deleted when the job ends; the rest is deleted `AUDITOR_STT_JOB_TTL_HOURS` after it ends (default 72; checked hourly), or at once by `DELETE /v1/jobs/{id}`. A `source_path` input lives on the caller's media volume and is never deleted by the service |
| Live requests | temporary file | The uploaded audio chunk | Written to a temporary file for the request and removed afterwards; the transcript goes back to the caller and is not stored |
| Logs | stdout of the commands and the service | Ids, counts, timings, error classes | By design no audio and no transcript text (module docstrings of `sync`, `build`, `lineage`, `eval`, the job runner); not audited line by line |

Outside the code and not covered anywhere in this document: backups and snapshots of the volumes, container layers that were committed by hand, copies made by the operator.

Pseudonymous is not anonymous. The `speaker_id` is the same for all recordings of one person, whoever holds the csv secret and user table can map it back to a person, and the voice is itself identifying.

## 3. Controls that exist in code

| Principle | What the code does | Where |
|---|---|---|
| Purpose limitation (structural) | Training reads only the ledger, which only `dataset sync`/`pull` fill from csv. The service code (`auditor_stt/serve`) does not import the dataset or training packages, and the dataset package does not import the service; there is no code path from `/inference` or the job store to the ledger. The two use different directory settings: `AUDITOR_STT_TRAIN_DATA_DIR` and `AUDITOR_STT_DATA_DIR`. `train` and `dataset build` never open the job store | `dataset/*`, `serve/*`, checked by import search |
| Minimisation of what is fetched | The sync client uses the export routes and the audio route only. It never calls `/api/admin/*`, which return emails and consent timestamps | `dataset/client.py` |
| Minimisation of what is stored | The ledger fields in section 2, nothing else. No email, no raw csv user id. The export's `speaker_id` is a pseudonym; a raw numeric id in an export is hashed with a local salt (`AUDITOR_STT_SPEAKER_SALT`) or dropped; email-like values are dropped | `dataset/sync.py` (`_resolve_speaker`), `dataset/ledger.py` |
| No external tracking | `report_to="none"` in training. No code pushes datasets or models anywhere: a search for `push_to_hub` and hub upload calls finds none | `training/train.py` |
| Transparency about lineage | Every dataset has a manifest and version; every run records the dataset version and manifest hash; the registry and `gate.json` carry the version | `dataset/build.py`, `training/lineage.py`, `serve/registry.py` |
| Deletion follows csv | A recording missing from the export is tombstoned on the next sync: audio file deleted first (unless shared), then text, hashes, speaker id and other content erased from the row; `PRAGMA secure_delete=ON` | `dataset/sync.py`, `dataset/ledger.py` |
| Guard against a broken export wiping the ledger | A sync refuses to tombstone more than 5 rows and more than half of a corpus without `--allow-mass-removal` | `dataset/sync.py` |
| Individual rights support | `dataset lineage --speaker` and `dataset purge --speaker` (section 6); `dataset prune` (section 8) | `dataset/lineage.py` |
| Evaluation output holds no personal text | `gate.json` and console output hold metrics only; no transcripts, no speaker ids (per-speaker results are a spread) | `training/gate.py` |
| Consumer audio is short-lived | Job upload copies and normalised audio deleted when the job ends; results expire by TTL; live audio only in a temporary file | `serve/jobs/runner.py`, `serve/jobs/store.py`, `serve/audio.py` |
| Offline operation | After the first download, `HF_HUB_OFFLINE=1` stops Hugging Face Hub requests. Sync and pull contact only the csv base URL | training image README |

The one link between the packages runs the other way: the evaluation code (`training/evaluate.py`) reuses the service's `ModelHost` and batch pipeline as code. It reads no job data.

The purpose separation is structural in the code, not enforced by the operating system. Someone with access to both directories can copy files between them, and if both settings point at the same directory tree nothing stops mixing. Use separate directories or volumes (section 9).

## 4. What the code does not do

- No encryption at rest and no file-permission setting; that is the job of the disk, the volume and the operator account.
- No access control on the CLI. Whoever can run the commands and read the directories has everything.
- No audit trail of who ran what. `purge` prints what it did and leaves `SUPERSEDED.json` (dataset version, time, reason, number of recordings; not the speaker), and `current.json` records `forced` but `promoted_by` is always `cli`. Keep your own record of erasure requests and of who acted.
- No consent check. Neither csv's export nor the sync looks at consent (csv side: open item 3).
- No per-recording lookup: `lineage` and `purge` work per speaker. A single deleted recording is tombstoned by sync, but the dataset versions that contain it keep it until they are superseded or pruned.
- No unlearning. A trained model cannot be edited to forget a speaker; the only options are retraining without the data or retiring the model.
- No automatic retention: nothing in the training system expires by itself, except that a run keeps at most two checkpoints while it trains.
- The long-form evaluation audio is a recording of speech by people who are not csv volunteers. The code reads it for evaluation only, never stores it in a ledger or dataset, writes only metrics, and deletes its temporary WAV. Whose voices it contains and on what basis it may be used for evaluation is for the owner to decide.

## 5. Model memorisation

A fine-tuned model was trained on the recordings. Whether it can reproduce training utterances or reveal that a person's data was used, and how that interacts with withdrawal, is not answered by this code (open item 7). Two facts from the code: intermediate checkpoints are full copies of the training state (`dataset prune --runs-dir` deletes them after a run is finished), and the merged `_merged_hf` checkpoint in an export directory is not needed for serving and is never copied into the registry.

## 6. Withdrawal and erasure

Order matters. A `purge` does not stop the next `dataset sync` from restoring recordings that csv still lists, so the deletion in csv comes first.

The commands below assume the default layout (run from the repository root; in the training container use the paths from the training image README, with `--models-dir /models`).

**Step 0. Find the speaker id, before the person is deleted in csv.** The training system never stores who a speaker is, so the id has to come from somewhere that knows:

- From the ledger, if you know one of the person's csv recording ids (`sqlite3 data/ledger.sqlite "SELECT DISTINCT speaker_id FROM recordings WHERE recording_id IN (123, 456)"`), or from the `manifest.json` of any dataset version (rows carry `recording_id` and `speaker_id`). The CLI has no command for this.
- Or from the csv side: the id is the first 24 hex characters of HMAC-SHA256(key = csv `SPEAKER_ID_SECRET`, message = csv user id). With the secret in the environment (read it from your secret store, not from the command line):

  ```
  node -e "const c=require('crypto');console.log(c.createHmac('sha256',process.env.SPEAKER_ID_SECRET).update(String(process.argv[1])).digest('hex').slice(0,24))" <csv user id>
  ```

  Only whoever has csv access holds the secret; the training system does not need it. The same computation exists in Python as `hash_speaker_id` in `auditor_stt/dataset/sync.py`.

After a recording has been tombstoned, the ledger no longer holds its speaker id.

**Step 1. Delete in csv.** The person deletes the account (`DELETE /api/me`, which removes the recordings and their audio files) or an administrator deletes it (`DELETE /api/admin/users/:id`). Note that `POST /api/me/anonymize` is not erasure: the recordings stay in csv without a user link, the next sync updates the ledger rows to speaker-less and keeps them as train rows, and dataset versions and lineage records already built keep the old pseudonym.

**Step 2. Sync.**

```
auditor-stt dataset sync --base-url https://csv.example.org --corpus-id 1
```

The recordings are tombstoned and their audio files deleted (unless another active recording has the identical audio). If many recordings vanish at once the mass-removal guard may stop the sync; check that the deletions are real, then add `--allow-mass-removal`. Sync each corpus the person contributed to.

**Step 3. See what used the speaker.**

```
auditor-stt dataset lineage --speaker <speaker id> --models-dir ./models
auditor-stt dataset lineage --speaker <speaker id> --models-dir ./models --json
```

It lists the dataset versions whose manifest names the speaker, with the number of recordings and the splits (a speaker who was only in `test` was scored on but not trained on), and the models and training runs under `--models-dir` whose metadata (`training_metadata.json`, registry `metadata.json`) names those dataset versions. It only sees datasets under `<data-dir>/datasets` and models whose metadata records a dataset version. Keep the output as your record: it holds counts and paths, no transcripts.

**Step 4. Purge.**

```
auditor-stt dataset purge --speaker <speaker id> --models-dir ./models          # dry run
auditor-stt dataset purge --speaker <speaker id> --models-dir ./models --yes
```

Without `--yes` nothing changes. With it: the speaker's active ledger rows are tombstoned and their audio files deleted (after Step 2 there may be nothing left to do here, and the counts say so), every dataset version whose manifest names the speaker is reduced to a `SUPERSEDED.json` marker (the data files, the embedded audio and the cached features are deleted; the marker holds the dataset version, time, reason and the number of recordings, not the speaker), and interrupted dataset builds are removed too because they may hold audio copies. It is safe to run again after an interruption. Other speakers in a superseded dataset version are affected too: the version is gone, and a new `dataset build` recreates a dataset from the ledger without the purged speaker.

**Step 5. Models are only reported.** The command lists the models and training runs that were trained on the superseded versions, and says "retrain or retire required". Nothing is modified. What to do, and how fast, is a decision for the owner and the DPO. The tools give these options:

- Retrain: `dataset build` (the tombstoned speaker is excluded automatically; new version), `train`, `export`, `eval`, `models register`, `models promote`, then switch the service (`POST /model`).
- Retire: promote another version, or switch the service to another model (`POST /model` with a `registry:` spec or an alias listed in `AUDITOR_STT_ALLOWED_MODELS`), and delete the model's directories by hand (runs, exports, the registry version; there is no registry delete command).

## 7. What `purge` does not reach

- Legacy `dataset pull` snapshots (`recordings.json` and audio) and datasets built with `dataset build --snapshot`.
- Datasets built to a custom `--out`: `lineage`, `purge` and `prune` scan `<data-dir>/datasets` only. Do not build elsewhere if you want these commands to see the data.
- Models already trained: training runs, checkpoints, exports (`_merged_hf`), registry versions, and the service if it is running one of them. Only reported, never modified.
- A single recording removed in csv while the speaker stays: it is tombstoned by sync, but datasets that already contain it keep it.
- Speakers who were anonymised in csv before the request (no id to search for; `POST /api/me/anonymize` leaves the recordings speaker-less in the ledger).
- Backups, volume snapshots, copies made by the operator, `--dump-predictions` files, and logs (which hold no text or audio, but do carry recording ids).
- Any Docker image or container that was committed with data inside. The training Dockerfiles copy only the code and no data.
- Disk-level remanence after deletion; `secure_delete` covers freed pages inside the SQLite file only.

## 8. Retention

Tooling:

```
auditor-stt dataset prune --keep-datasets 2 --models-dir ./models                 # dry run
auditor-stt dataset prune --keep-datasets 2 --models-dir ./models --yes
auditor-stt dataset prune --runs-dir models/runs --yes                             # checkpoints of finished runs
```

- `--keep-datasets N` keeps the newest N built dataset versions (by `created_at`) and deletes the rest, except versions that a model under `--models-dir` names; `--include-referenced` deletes those too and loses the record of what such a model was trained on. Without `--yes` it only shows what would be deleted.
- `--runs-dir DIR` deletes the `checkpoint-N` directories inside finished training runs (runs with a `training_metadata.json`) and keeps the final model. Checkpoints in unfinished runs are left alone.
- At least one of `--keep-datasets` and `--runs-dir` is required.

No retention period is set anywhere in the code and none is proposed here. Periods to be decided by the owner:

- Ledger rows and the normalised audio store (active recordings)
- Tombstones (the code keeps them indefinitely; they hold only ids and timestamps)
- Dataset versions, and how many to keep
- Training runs and checkpoints
- Exports, including `_merged_hf`
- Registered model versions, and `gate.json` and training metadata
- `--dump-predictions` output
- Service job results (the 72 hour default is a configured default, not a decided period)
- Logs, backups and their rotation
- How long records of erasure requests are kept

## 9. Storage rules

- Local only. Datasets, checkpoints, exports and the ledger stay on the operator's own machine or server. Nothing in the code pushes to a hub; do not add it, and do not upload datasets or fine-tuned models anywhere without a review of the training data's provenance.
- Once the base model is downloaded, set `HF_HUB_OFFLINE=1` (`-e HF_HUB_OFFLINE=1` in `docker run`) so nothing is fetched. The first run, the `--merge-lora` export and the `eval` baseline alias need the network unless the weights are already cached.
- Encrypt the disk (or volume) that holds the data directory, the models directory and Docker's volumes, and limit access to the operator account. The code does neither.
- Keep the data outside git. The repository `.gitignore` covers `/data/` and `/models/` at the repository root only; `./data` created inside `python-packages/liturgos-auditor-stt/` is not ignored, and `.wav` files are not ignored anywhere. Run from the repository root or use an absolute path outside the repository (`--data-dir`, `AUDITOR_STT_TRAIN_DATA_DIR`).
- Keep the training data apart from the service. Do not point `AUDITOR_STT_DATA_DIR` (job store) and `AUDITOR_STT_TRAIN_DATA_DIR` at the same tree, and keep the training data, runs and exports out of the service's `auditor-stt-models` and `auditor-stt-data` volumes. Only the CT2 export that `models register` copies into the registry belongs in the volume the service reads.
- Container mounts: mount only what a command needs. `sync`, `build`, `lineage`, `purge`, `prune` open the ledger and the data directory read-write (the ledger switches to WAL mode and writes a schema row on opening, so a read-only mount is not expected to work). `train` needs the dataset directory (it writes cache files there) and its output directory; `export` the run and the output directory; `eval` the dataset, the export and, for the long-form check, the private audio and reference (read-only is fine); `models` the registry. A `-v` mount of the private long-form directory keeps that audio out of the image.
- Secrets: pass the csv token with `-e CSV_ADMIN_TOKEN` (value taken from the host environment), never on the command line or in an image. csv's `SPEAKER_ID_SECRET` does not belong on the training host except briefly for Step 0.
- Do not `docker commit` a container that has run with data.

## 10. Roles

This document does not say who is the controller and who is a processor. Questions for the owner and the DPO:

| Question | Why it matters |
|---|---|
| Which organisation determines the purposes and means of the training (the controller)? Is it the same organisation that runs csv and to which csv's privacy text will refer? | Contributors are told about one organisation ([ORGANIZATION NAME] in csv's texts); the training system must be under the same controller or have its own basis and notice |
| Is the maintainer acting as an employee or volunteer of the controller, or as an external party who processes on its behalf? | An external processor needs a data processing agreement |
| Who else touches the data: hosting of csv, of the training machine, backups, any support person? | Each is a processor or sub-processor to be listed and assessed |
| Is the Hugging Face Hub a recipient? | This code downloads public weights and sends no personal data; the owner should confirm that this is how it is operated |
| Who answers a data subject's request (access, erasure, objection) and within which process? | The tools in section 6 need an owner |

## 11. Open items that block real data

The project plan's position is that a DPO review comes before the first real sync. Until these are settled, do not run `dataset sync` against a corpus that holds real recordings.

1. **csv privacy policy and terms are templates.** `PrivacyPolicy.jsx` has `[DATE]`, `[ORGANIZATION NAME]`, `[CONTACT EMAIL]` and `[ORGANIZATION ADDRESS]`. `TermsOfService.jsx` has `[DATE]`, `[ORGANIZATION NAME]` (several times), `[LICENSE TYPE, e.g., CC0, CC-BY, or custom license]`, `[JURISDICTION]`, `[CONTACT EMAIL]` and `[ORGANIZATION ADDRESS]`. No controller is named, no data licence is chosen, and no lawful basis is stated. Retention is described only as "as long as your account is active or as needed to provide the Service", and the privacy policy adds that anonymised recordings "may be retained indefinitely as part of training datasets".
2. **No consent version is stored per recording.** csv stores `users.terms_accepted_at` and `users.recording_consent_at` (timestamps only). The consent wording is hard-coded in `Record.jsx`, so which text a given recording was made under cannot be shown later. Consent wording is for the owner and DPO to write; this document proposes none.
3. **csv's export ignores withdrawal of recording consent.** `DELETE /api/me/consent/recording` sets `recording_consent_at` to NULL and touches nothing else; the export never looks at it, so a person who withdraws consent still has all recordings exported and synced. The consent gate also says that once anonymised or included in released datasets, recordings cannot be withdrawn; whether that is acceptable is for the DPO.
4. **csv's `/uploads` audio URLs are unauthenticated** (`express.static`). Anyone who has a URL can fetch the audio. `GET /api/recording/:id` and `GET /api/validation/flagged` return `file_path` (and `user_id`) to any logged-in user. The sync client downloads through this route. More csv findings are in [crowd-source-voice-contract.md](crowd-source-voice-contract.md).
5. **Purpose as told to contributors.** Both csv texts describe "speech recognition and synthesis datasets", sharing with researchers and public release. Whether training a model for this organisation's own transcription service is within what contributors were told, and whether the notice must name it, is for the DPO.
6. **Special-category question.** Whether voice recordings of religious text, or a volunteer's participation in a church project, touch data revealing religious belief or affiliation (GDPR Art. 9), and what that requires. Related: whether a voice recording is biometric data in this use (it is not used to identify anyone here). The DPO decides.
7. **Model memorisation and withdrawal.** Whether a trained model can reproduce training utterances or reveal membership, and how withdrawal or erasure applies to a model trained on the data. The code can only report affected models and leave retrain-or-retire to the owner.
8. **DPIA.** Whether a data protection impact assessment is required before processing.
9. **Roles and lawful basis** (section 10): controller, processors, and the basis for training on csv data.
10. **Pseudonymisation wording.** csv's consent text says recordings are associated with "anonymized IDs". The `speaker_id` is a stable pseudonym derived with a secret held by csv's operator, and a voice is identifying; whether "anonymized" is an accurate description is for the DPO.
11. **Minimisation of what is kept.** `quality_score` is stored in the ledger and copied into datasets but nothing in the training code uses it; the owner may want to drop it.
