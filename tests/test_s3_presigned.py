from __future__ import annotations

import json
from pathlib import Path

from scripts.build_step8_notebook import build
from scripts.generate_presigned_config import run_keys
from src.training import S3ArtifactUploader


class _Response:
    def __init__(self, body: bytes = b"") -> None:
        self.body = body

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self): return self.body


def test_presigned_uploader_uses_object_scoped_put_and_get(tmp_path, monkeypatch):
    root = "thesis/gc-lstm-ghostnet/run1"
    upload_key = f"{root}/checkpoints/last_checkpoint.pt"
    config = {"bucket": "bucket", "resume_required": True,
              "uploads": {upload_key: "https://put.invalid"},
              "downloads": {upload_key: "https://get.invalid"}}
    config_path = tmp_path / "s3.json"; config_path.write_text(json.dumps(config))
    monkeypatch.setenv("S3_PRESIGNED_CONFIG_PATH", str(config_path))
    calls = []

    def fake_urlopen(request, timeout=0):
        calls.append((getattr(request, "method", "GET"), str(getattr(request, "full_url", request))))
        return _Response(b"downloaded")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    source = tmp_path / "source.pt"; source.write_bytes(b"checkpoint")
    target = tmp_path / "target.pt"
    uploader = S3ArtifactUploader(True, None, root, None, 0, True)
    assert uploader.resume_required is True
    assert uploader.upload(source, "checkpoints/last_checkpoint.pt")
    assert uploader.download("checkpoints/last_checkpoint.pt", target)
    assert target.read_bytes() == b"downloaded"
    assert calls == [("PUT", "https://put.invalid"), ("GET", "https://get.invalid")]


def test_manifest_and_notebook_cover_cpu_resume_contract(tmp_path):
    keys = run_keys("thesis", "run1", 100)
    assert "thesis/active_run.json" in keys
    assert "thesis/gc-lstm-ghostnet/run1/checkpoints/epoch_100.pt" in keys
    assert "thesis/gc-lstm-ghostnet/run1/checkpoints/last_checkpoint.pt" in keys
    config = {"bucket": "bucket", "s3_prefix": "thesis", "run_id": "run1", "aws_region": "us-east-1",
              "uploads": {}, "downloads": {}, "resume_required": False}
    config_path = tmp_path / "s3.json"; config_path.write_text(json.dumps(config))
    notebook = build(config_path)
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "PRESIGNED_CONFIG_B64" in source
    assert '"--device", "cpu"' in source
    assert '"--resume", "auto"' in source
    assert "--upload-checkpoints-to-s3" in source
    assert "AWS_SECRET_ACCESS_KEY=" not in source
