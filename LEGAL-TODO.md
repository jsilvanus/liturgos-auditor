# Legal and compliance items still open

Status as of 2026-09-21. This is a maintainer's checklist, not legal advice. The code side
of data protection is built (see [docs/data-protection.md](docs/data-protection.md)); what
follows cannot be settled in code and needs the owner and a data protection officer (DPO).

**Do not run `auditor-stt dataset sync` against a corpus that holds real recordings until
items 1 to 8 are settled.** Nothing in this repository has been run on real speaker data;
all training-chain testing used synthetic recordings.

## Blocks real data

1. **crowd-source-voice privacy policy and terms are unfilled templates.** No controller
   is named, no data licence is chosen (`[LICENSE TYPE ...]`), no lawful basis is stated, and
   contact, jurisdiction and date are placeholders. Retention is only "as long as your
   account is active", and anonymised recordings "may be retained indefinitely".
2. **Consent is not versioned per recording.** Only `terms_accepted_at` and
   `recording_consent_at` timestamps are stored; the wording lives in `Record.jsx`, so it
   cannot later be shown which text a recording was made under.
3. **Withdrawal of recording consent is not honoured by the export.**
   `DELETE /api/me/consent/recording` clears a timestamp and nothing else, so a person who
   withdraws still has all recordings exported and synced. The consent gate also says
   anonymised or released recordings cannot be withdrawn; whether that is acceptable is a
   DPO question.
4. **Purpose as told to contributors.** The texts promise "speech recognition and synthesis
   datasets" and research or public release. Whether training this organisation's own
   transcription model is covered, and whether the notice must say so, is open.
5. **Special-category data.** Whether recordings of people reading religious text, or taking
   part in a church project, reveal religious belief or affiliation (GDPR Art. 9).
6. **DPIA.** Whether a data protection impact assessment is needed before processing.
7. **Roles and lawful basis.** Who is controller and who are processors, and on what basis
   the training happens.
8. **Models and erasure.** Whether a fine-tuned model can reproduce or reveal training
   utterances, and what withdrawal means for a model already trained. The tooling reports
   affected models (`dataset lineage`, `dataset purge`) but retrain-or-retire is a
   decision for the owner and DPO.

## Decisions the owner still has to make

- Retention periods for the ledger, audio store, dataset versions, checkpoints and
  registered models (the tooling exists, `dataset prune`; **no periods are set**).
- Whether "anonymized IDs" in csv's consent text is accurate for a stable pseudonym derived
  with a secret (`speaker_id`).
- Whether to keep `quality_score` in the ledger at all (stored, currently unused).
- Whether any consumer audio (lcyt live captions, saarnavideo sermons) may ever be used
  for training. Today it cannot: job audio is deleted with the job and nothing feeds the
  training ledger. Changing that needs its own lawful basis.

## Security items in crowd-source-voice that affect the same data

- With the local storage driver `/uploads` audio is served without authentication (the
  sync itself no longer uses it: it downloads through a token-gated route). Use the S3
  driver with a private bucket for real data.
- Uploads accept a file named `x.html` sent with an audio MIME type and serve it back as
  HTML from `/uploads` (stored cross-site scripting on csv's origin).
- `GET /api/recording/:id` and `GET /api/validation/flagged` return `user_id` and
  `file_path` to any logged-in user.
- `server/middleware/auth.js` falls back to a hard-coded `JWT_SECRET` if the variable is
  unset: confirm production sets it.
- Before the first sync, set `SPEAKER_ID_SALT` (long, random, never changed afterwards,
  because speaker ids derive from it; csv's `.env.example` ships a public placeholder) and
  `EXPORT_API_TOKEN` in csv's deployment. The speaker id is a salted hash of the
  contributor's **email**, so the salt is what keeps it private.

Details and the export contract: [docs/crowd-source-voice-contract.md](docs/crowd-source-voice-contract.md).
