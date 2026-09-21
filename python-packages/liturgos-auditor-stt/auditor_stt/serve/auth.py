"""Optional bearer-token auth for the service (AUDITOR_STT_API_KEY).

Off unless a key is configured: lcyt's whisper-http adapter cannot send headers,
so the default deployment relies on network isolation. lcyt's OpenAI adapter can
send `Authorization: Bearer <key>`, which is what the key protects. Routers take
`dependencies=[Depends(require_api_key)]`; /health stays outside it on purpose so
orchestrator probes need no secret.
"""

import hmac

from fastapi import HTTPException, Request


def require_api_key(request: Request) -> None:
    key = request.app.state.api_key
    if not key:
        return
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    # Bytes, not str: compare_digest rejects non-ASCII str, which would turn a junk header into a 500.
    if scheme.lower() == "bearer" and hmac.compare_digest(token.strip().encode(), key.encode()):
        return
    raise HTTPException(
        status_code=401,
        detail="Invalid or missing API key",
        headers={"WWW-Authenticate": "Bearer"},
    )
