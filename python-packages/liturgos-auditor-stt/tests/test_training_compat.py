import types

import pytest
from packaging.version import Version

from auditor_stt.training import compat


def test_save_processor_writes_the_processor_and_the_feature_extractor_files(tmp_path):
    saved = []
    processor = types.SimpleNamespace(
        save_pretrained=lambda path: saved.append(("processor", path)),
        feature_extractor=types.SimpleNamespace(save_pretrained=lambda path: saved.append(("feature_extractor", path))),
    )
    compat.save_processor(processor, tmp_path)
    assert saved == [("processor", str(tmp_path)), ("feature_extractor", str(tmp_path))]


@pytest.mark.parametrize("version, expected", [
    ("4.46.0", {"warmup_ratio": 0.05}),
    ("4.57.1", {"warmup_ratio": 0.05}),
    ("5.0.0", {"warmup_steps": 0.05}),
    ("5.17.0", {"warmup_steps": 0.05}),
])
def test_warmup_kwargs_follow_the_transformers_version(monkeypatch, version, expected):
    monkeypatch.setattr(compat, "_transformers_version", lambda: Version(version))
    assert compat.warmup_kwargs(0.05) == expected
