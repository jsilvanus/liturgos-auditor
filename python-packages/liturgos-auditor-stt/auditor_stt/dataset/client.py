"""HTTP client for the crowd-source-voice admin export API.

Auth is a bearer token: either an ordinary user JWT for an admin-role account
(crowd-source-voice's original scheme) or its long-lived export token, which
newer versions also accept on /api/export*. Pass it in, e.g. read from an env
var by the caller. Only the export endpoints and the audio route are used;
never /api/admin/*, which exposes emails and consent records.
"""

import httpx


def _is_empty_listing(resp):
    """crowd-source-voice answers 404 {"error": "No recordings found for export"} when nothing qualifies."""
    if resp.status_code != 404:
        return False
    try:
        return "no recordings found" in str(resp.json().get("error", "")).lower()
    except (ValueError, AttributeError):
        return False


class CrowdSourceVoiceClient:
    def __init__(self, base_url, token, transport=None):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(transport=transport, timeout=60.0, headers={"Authorization": f"Bearer {token}"})

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def get_export(self, corpus_id):
        """GET /api/export?corpus_id=&format=json. Never passes include_all — training only ever consumes the validated export."""
        resp = self._client.get(f"{self.base_url}/api/export", params={"corpus_id": corpus_id, "format": "json"})
        resp.raise_for_status()
        return resp.json()

    def get_export_or_empty(self, corpus_id):
        """Like get_export, but an empty listing (404 "No recordings found") returns an empty export (`corpus: None`) instead of raising.

        Sync needs the distinction: an empty corpus is a valid state (every
        recording deleted upstream), a failed request is not.
        """
        resp = self._client.get(f"{self.base_url}/api/export", params={"corpus_id": corpus_id, "format": "json"})
        if _is_empty_listing(resp):
            return {"corpus": None, "total_recordings": 0, "recordings": []}
        resp.raise_for_status()
        return resp.json()

    def get_manifest(self, corpus_id):
        """GET /api/export/manifest?corpus_id= — source file paths + speaker_id."""
        resp = self._client.get(f"{self.base_url}/api/export/manifest", params={"corpus_id": corpus_id})
        resp.raise_for_status()
        return resp.json()

    def get_manifest_or_empty(self, corpus_id):
        """Like get_manifest, but an empty listing returns an empty manifest (a recording can vanish between the export and manifest calls)."""
        resp = self._client.get(f"{self.base_url}/api/export/manifest", params={"corpus_id": corpus_id})
        if _is_empty_listing(resp):
            return {"total": 0, "files": []}
        resp.raise_for_status()
        return resp.json()

    def download_audio(self, source_path):
        """Fetch one recording's audio bytes.

        The path is relative to the base URL. For a current export it is the
        row's `audio_url` (`/api/export/audio/<recording_id>`), a token-gated
        route that streams the file from csv's storage (disk or S3); the bearer
        header of this client is what authorises it. An old export points at the
        static `/uploads/...` route instead, which needs no token.
        """
        resp = self._client.get(f"{self.base_url}/{source_path.lstrip('/')}")
        resp.raise_for_status()
        return resp.content
