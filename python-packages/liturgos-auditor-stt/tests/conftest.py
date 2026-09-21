import pytest

from auditor_stt.serve import model


@pytest.fixture(autouse=True)
def _fresh_cuda_state():
    """ModelHost remembers a failed CUDA attempt for the life of the process; tests must not inherit it."""
    model.reset_cuda_state()
    yield
    model.reset_cuda_state()
