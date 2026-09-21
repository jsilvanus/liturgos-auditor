"""Batch full-file transcription jobs, checkpointed chunk by chunk.

Pure logic lives here (WAV access, chunk planning, the on-disk job store,
result assembly); nothing in these modules touches HTTP or the model.
"""
