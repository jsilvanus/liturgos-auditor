# crowd-source-voice export contract

What `auditor-stt dataset sync` (and the older `dataset pull`) expects from crowd-source-voice (csv), and what csv provides. "v2" is this repository's name for the csv export that carries `recording_id`, `audio_url` and `speaker_id`, serves the audio through a token-gated route and accepts a read-only export token (the auditor tests call it `v2`); "legacy" is the export without them.

Where each part comes from in the csv repository: `speaker_id` was added by csv PR #4 (`server/utils/speakerId.js`) and S3 storage by PR #5 (`server/utils/storage.js`); `recording_id`, `audio_url`, the export token (`server/middleware/exportAuth.js`) and the audio route (`GET /api/export/audio/:recordingId`) are on the branch `export-audio-route` (not merged at the time of writing). The auditor side is `auditor_stt/dataset/`.

How the data is used and protected afterwards is in [data-protection.md](data-protection.md). The pipeline is in [training-pipeline.md](training-pipeline.md).

## 1. Routes

All four are `GET` under `/api/export` and accept `Authorization: Bearer <token>` where the token is an admin JWT or csv's `EXPORT_API_TOKEN`.

| Route | Query | Used by the sync client |
|---|---|---|
| `/api/export` | `corpus_id` (required), `format` (`csv` default, or `json`), `include_all` | Yes, with `format=json`; never `include_all` |
| `/api/export/manifest` | `corpus_id` (required) | Only against a legacy export (section 4) |
| `/api/export/stats` | `corpus_id` (optional) | No |
| `/api/export/audio/:recordingId` | `include_all` (admin JWT only) | Yes: this is the row's `audio_url` |

Audio is streamed by `/api/export/audio/:recordingId` from whichever storage driver csv uses (local disk, or S3/MinIO with `STORAGE_DRIVER=s3`), so the client needs only this API and one bearer token, never direct access to the uploads folder or the bucket. The client fetches `<base-url><audio_url>` with the same bearer header. The route answers 200 with the file bytes and `Cache-Control: private, no-store`; 404 `{"error":"Recording not found"}` with an identical body when the recording does not exist, is not validated, or its file is missing; 400 for an id that is not a positive integer; 403 for the export token with `include_all=true`; 401 without a valid token.

Which recordings are exported (and which audio the route serves to the export token): those with a quality score of at least 4.0 and at least 2 validations (`MIN_SCORE_THRESHOLD`, `MIN_VALIDATIONS` in `export.js`). `include_all=true` drops both filters, is honoured for an admin JWT only, and is answered with 403 for the export token. The manifest always applies the filters.

### 1.1 `GET /api/export?corpus_id=N&format=json`

Top level:

| Field | Meaning |
|---|---|
| `corpus` | `{id, name, language, type}`; `type` is `text` or `music`. The client accepts `text` only |
| `total_recordings` | Number of rows |
| `recordings` | Array of rows, ordered by recording id |

Each row:

| Field | Meaning |
|---|---|
| `file` | Positional name (`0001.wav`, ...). Shifts between calls as recordings qualify; not an identifier |
| `recording_id` | csv's `recordings.id`; the stable identifier. The client uses this as the ledger key |
| `original_path` | The raw stored value: a storage key such as `audio/<uuid>.<ext>` (csv with the storage driver, PR #5) or a legacy `/uploads/audio/<uuid>.<ext>` path. Kept for backward compatibility; it is not a fetchable address, and the sync client does not use it when `audio_url` is present |
| `audio_url` | `/api/export/audio/<recording_id>`: where to fetch the audio (section 1) |
| `speaker_id` | Pseudonym (section 3), or `null` |
| `text` | The prompt text that was read (`notation` instead of `text` for music corpora) |
| `duration` | Seconds, as reported by the browser; nullable and not verified server-side |
| `quality_score` | Average validation score, float |
| `validation_count` | Integer |

Not in the export: email, password hash, consent timestamps, raw user id.

Errors the client treats specially: `404 {"error": "No recordings found for export"}` means an empty listing (the client treats it as an empty corpus). Other errors, including `404 {"error": "Corpus not found"}`, `400` (no `corpus_id`), `401` and `403`, abort the sync with a message and change nothing.

With `format=csv` (default) the columns are `file,text,duration,quality_score` (`notation` for music corpora). The auditor does not use it.

### 1.2 `GET /api/export/manifest?corpus_id=N`

`{total, files: [{id, recording_id, source_path, export_name, speaker_id, text}]}`. `id` and `recording_id` have the same value; `source_path` is the file path; `export_name` is the positional name that matches `file` in the export. Only the legacy path of the client reads it.

### 1.3 `GET /api/export/stats`

Rows of `{corpus_id, corpus_name, type, total_recordings, exportable_recordings, total_duration_seconds}`. Not used by the auditor.

## 2. What the sync client does with a row

| Check | Behaviour |
|---|---|
| Corpus type | Must be `text`; otherwise `Corpus N ... is type 'music'` and no change |
| Text | Normalised (NFC, literal `\n` and line breaks to spaces, whitespace collapsed); an empty text is skipped (`empty_text`) |
| Audio | Downloaded from `audio_url` (falling back to `original_path`), converted with ffmpeg to 16 kHz mono PCM16 WAV |
| Length | 0.5 to 30 s, measured on the converted audio. csv's own `duration` is client-supplied and never used for this. csv's README allows recordings up to 120 s, so longer ones are skipped (`duration_out_of_range`). A rejected recording is remembered in the ledger's `rejected` table (id, reason, audio path only) and not downloaded again on later syncs until its audio path changes or it leaves the export |
| Speaker | Section 3 |
| Language | The corpus's `language` field is not used; the training language is a flag of `train` (default `fi`) |
| Identity | `recording_id`. A recording missing from a later export is treated as deleted (there are no tombstones or deleted-id lists in csv) |
| Timeouts | 60 s per request; the export is one response with no pagination |

## 3. `speaker_id`

csv derives it in `server/utils/speakerId.js` (`computeSpeakerId`, csv PR #4):

```
first      = SHA-256( SPEAKER_ID_SALT + ":" + lowercase(trim(email)) )
speaker_id = SHA-256( SPEAKER_ID_SALT + ":" + first )          -> 64 lowercase hex characters
```

- It is derived from the contributor's **email address**, which is personal data; the id is a salted one-way hash of it, never the email itself. The same email always gives the same id, on any corpus and export, as long as the salt does not change.
- `null` when the recording has no user or the user has no email (an anonymised recording).
- `SPEAKER_ID_SALT` unset means an empty salt: csv only logs a warning at startup, and the ids are then a plain double hash that anyone can compute for a known email address. Set it in every environment (see section 5).
- It is a pseudonym, not anonymisation (data-protection.md).

A one-line check of the derivation, with the salt in the environment:

```
node -e "const c=require('crypto');const h=s=>c.createHash('sha256').update(s).digest('hex');const s=process.env.SPEAKER_ID_SALT||'';const e=process.argv[1].trim().toLowerCase();console.log(h(s+':'+h(s+':'+e)))" someone@example.org
```

### What the sync client stores

- A string that is neither all digits nor contains `@` is taken to be csv's pseudonym and stored as given (also when a salt is set).
- A JSON integer, or an all-digit string of 1 to 18 characters, is taken to be a raw csv user id, which the current csv export never sends. It is hashed with `AUDITOR_STT_SPEAKER_SALT` (a local HMAC-SHA256, truncated to 24 hex characters; this is not csv's derivation) or, when the variable is not set, dropped (`raw_speaker_id_dropped`, with a warning in the log). The variable name can be changed with `--speaker-salt-env`. With the current csv export the salt is never used.
- Anything containing `@`, and any value that is not a string or an integer, is dropped (`unusable_speaker_id_dropped`).
- `null` is stored as speaker-less; such rows are train-only in datasets.

Raw csv user ids are never persisted by the training system. `dataset pull`, the legacy path, does not filter and writes the export's `speaker_id` into `recordings.json` as it came.

## 4. Legacy behaviour the client still supports

If any row of the export lacks `recording_id`, the client treats the whole listing as legacy:

1. It fetches `/api/export/manifest` and pairs rows with manifest files by the positional name (`file` and `export_name`).
2. It verifies the pairing: same number of rows and files, every row's name present in the manifest, and the manifest's `text` (when present) equal to the row's after normalisation.
3. On a mismatch it fetches both again once (a recording can cross the validation threshold between the two calls and shift every later name, which would pair one recording's text with another's audio). If they still disagree it aborts with `Export and manifest ... still disagree after a refetch` and stores nothing.
4. It then takes the identity from the manifest's `id`, the audio path from `source_path`, and the speaker from the row, else from the manifest.

The legacy path only makes sense against an old csv: since csv PR #5 `source_path` is a raw storage key, not something a client can fetch, so a csv that has that change but not the audio route cannot be synced at all. `dataset pull` (always the manifest pairing, positional names and the same verification) has the same limitation and should be regarded as obsolete.

## 5. csv environment variables

Set in csv's `.env` (see csv's `.env.example` and README):

| Variable | Purpose | Notes |
|---|---|---|
| `SPEAKER_ID_SALT` | Salt for the `speaker_id` derivation (csv PR #4) | Use a long random value (`node -e "console.log(require('crypto').randomBytes(32).toString('hex'))"`) and set it in EVERY environment: unset means an empty salt and guessable ids, and csv only warns at startup. csv's `.env.example` ships a public placeholder value and its prod/staging examples ship `change-me`: replace them. Do not change it after data was exported: every speaker gets a new id, so splits built on earlier exports no longer match, the next sync updates every ledger row to the new ids, and the old ids in earlier dataset versions and lineage records no longer match the ledger. Keep it private: with the salt and the user table anyone can map a pseudonym to a person. The training system does not need it |
| `STORAGE_DRIVER` (+ `S3_*`) | Where recordings are stored: `local` (default, `uploads/`) or `s3` (S3/MinIO bucket; csv PR #5) | The audio route reads from whichever is configured. With `s3` and a private bucket, nothing is served from `/uploads` and no presigned URL leaves csv |
| `EXPORT_API_TOKEN` | Long-lived, read-only bearer token for the three export GET routes | Off when unset or empty (then only admin JWTs work). Cannot use `include_all=true` (403). Grants nothing else: it is not a JWT and is useless on every other route. Use a long random value. Rotating it is safe: change the value, restart csv, update the clients |

## 6. How the sync client authenticates

- Header: `Authorization: Bearer <value>`, where the value is read from the environment variable named by `--token-env` (default `CSV_ADMIN_TOKEN`). The name is historical: the variable is only a bearer holder. It works with either an admin JWT (an admin account's login token, valid 7 days) or the export token. Use the export token for unattended syncing.
- If the variable is not set the command stops with `Env var CSV_ADMIN_TOKEN is not set`.
- The client calls only `/api/export` (json), `/api/export/manifest` (legacy only) and `/api/export/audio/:recordingId`. It never calls `/api/admin/*`, which return emails and consent timestamps.
- The same bearer header is sent on the audio download, and csv requires it there. The export token can only read validated recordings; it cannot use `include_all`.
- `AUDITOR_STT_SPEAKER_SALT` (section 3) is a separate variable for pseudonymising raw ids, not for authentication.

## 7. Items for the csv owner (not changed)

Findings from reading the csv code (GitHub `main` plus the `export-audio-route` branch). Several bear on whether real data may flow (data-protection.md, section 11).

| # | Finding | Evidence | Effect here |
|---|---|---|---|
| 1 | Withdrawal of recording consent is not honoured by the export. `DELETE /api/me/consent/recording` only sets `users.recording_consent_at` to NULL; the export never reads it | `server/routes/user.js`, `server/routes/export.js` | A person who withdraws consent keeps having all recordings exported and synced |
| 2 | `/uploads` is served without authentication (local storage driver), and the export still returns the raw stored path as `original_path` | `server/index.js`: `app.use('/uploads', express.static(...))`; `server/routes/export.js` | The sync client no longer uses this route (it uses the token-gated audio route). With `STORAGE_DRIVER=local`, anyone who knows a recording's path can still fetch it without the validation check, so use the S3 driver with a private bucket for real data |
| 3 | `language` is free text (`VARCHAR(50)`, only `notEmpty().trim()`), for example `English`, not a code | `server/db/migrate.js`, `server/routes/corpus.js` | The export's `corpus.language` cannot be used as a Whisper language code, so the auditor ignores it |
| 4 | The privacy policy and terms pages are templates with placeholders: `[DATE]`, `[ORGANIZATION NAME]`, `[CONTACT EMAIL]`, `[ORGANIZATION ADDRESS]`, `[JURISDICTION]`, `[LICENSE TYPE, e.g., CC0, CC-BY, or custom license]`. No data licence is chosen; the consent text has no stored version | `client/src/pages/PrivacyPolicy.jsx`, `TermsOfService.jsx`, `Record.jsx`, `users` table | Blocks real data (data-protection.md) |
| 5 | `GET /api/validation/flagged` (comment: "for admin review") and `GET /api/recording/:id` require a login but not the admin role, and return `r.*`, including `user_id` and `file_path`, to any logged-in user | `server/routes/validation.js`, `server/routes/recording.js` | Any account holder can learn the user id and audio URL of any recording |
| 6 | `JWT_SECRET` falls back to a hard-coded string when the environment variable is unset | `server/middleware/auth.js`: `process.env.JWT_SECRET \|\| 'your-secret-key-change-in-production'` | If csv runs without `JWT_SECRET`, anyone who knows that string can mint a token for any user id, including an admin, and reach every admin route (and thereby the export) |
| 7 | Upload type check is weak: a file is accepted when its client-declared MIME type is an allowed audio type OR its name ends in `.wav`, and the stored key keeps `path.extname(file.originalname)`. A file named `x.html` uploaded with MIME type `audio/wav` is stored as `audio/<uuid>.html` and, with the local driver, served by the public `/uploads` static mount as HTML | `server/routes/recording.js` (`fileFilter`), `server/utils/storage.js` (`createUploadMiddleware`), `server/index.js` | Stored cross-site scripting on csv's own origin, exploitable by any account that can upload. The export audio route is not affected (fixed content types, `X-Content-Type-Options: nosniff`) |
| 8 | The global error handler answers 500s with `err.message`, so database and driver errors can leak details (table names, paths, bucket names) on every route except the new audio route, which returns a fixed message | `server/index.js` | Informational; worth fixing before csv is exposed |
| 9 | `.env.example` ships `SPEAKER_ID_SALT=change-this-salt-in-production` (and the prod/staging examples `change-me`); with that value, or none, the speaker ids can be recomputed from a known email address | `.env.example`, `.env.prod.example`, `.env.staging.example`, `server/utils/speakerId.js` | The pseudonym is only as private as the salt; see section 5 |

Other observations, less important:

- Deletion is silent: csv hard-deletes and has no tombstones, deleted-id list or `since` parameter, so the auditor infers deletions from absence and guards against a broken export with the mass-removal limit.
- `DELETE /api/me` and the admin user delete swallow errors when removing audio files (`.catch(() => {})`), so a failed `unlink` leaves a file on the csv disk after its database row is gone.
- `POST /api/me/anonymize` sets `recordings.user_id` to NULL and keeps the recordings. The schema also has `ON DELETE SET NULL` on `recordings.user_id`.
- `duration` is supplied by the browser (`duration || null`) and is not checked on upload; csv's README says 0.5 to 120 s is enforced before submission, in the browser.
