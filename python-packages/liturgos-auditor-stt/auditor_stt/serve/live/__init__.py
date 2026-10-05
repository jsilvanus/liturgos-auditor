"""Live transcription, pull type: the service pulls a stream and publishes its text as Server-Sent Events.

    POST   /v1/live               start a session on an rtsp:// or srt:// source
    GET    /v1/live               sessions
    GET    /v1/live/{id}          status
    GET    /v1/live/{id}/events   SSE: transcript, status and error events
    DELETE /v1/live/{id}          stop

Sessions are not durable. A restart ends them; the client starts a new one.
"""
