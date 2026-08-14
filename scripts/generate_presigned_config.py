"""Generate short-lived, object-scoped S3 URLs for one GhostNet Kaggle session."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

AWS_ENV_NAMES = (
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_REGION", "AWS_DEFAULT_REGION",
)

ARTIFACT_FILES = (
    "history.json", "test_metrics.json", "summary_metrics.csv",
    "confusion_matrix.csv", "confusion_matrix.png", "accuracy_curve.png",
    "loss_curve.png", "auc_curve.png", "y_true.npy", "y_prob.npy",
    "label_mapping.json", "run_config.json", "sample_manifest.json",
    "label_schema_audit.json", "preprocessing.json", "preprocessor.joblib",
    "final_model_epoch_100.pt",
)


def normalize_aws_environment() -> None:
    for name in AWS_ENV_NAMES:
        value = os.environ.get(name)
        if value is not None:
            os.environ[name] = value.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--s3-prefix", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--expires", type=int, default=86400)
    parser.add_argument("--region")
    return parser.parse_args()


def read_active(client, bucket: str, prefix: str) -> dict:
    try:
        response = client.get_object(Bucket=bucket, Key=f"{prefix}/active_run.json")
        return json.loads(response["Body"].read().decode("utf-8"))
    except Exception as exc:
        response = getattr(exc, "response", None) or {}
        code = response.get("Error", {}).get("Code")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return {}
        raise


def resolve_run(client, bucket: str, prefix: str, requested: str | None) -> tuple[str, dict]:
    active = read_active(client, bucket, prefix)
    if requested:
        return requested, active if active.get("run_id") == requested else {}
    if (active.get("status") in {"running", "paused", "ready_for_report"}
            and active.get("contract_version") == "exhaustive-label-v1"):
        return str(active["run_id"]), active
    return f"ghostnet_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}", {}


def run_keys(prefix: str, run_id: str, epochs: int) -> set[str]:
    root = f"{prefix}/gc-lstm-ghostnet/{run_id}"
    keys = {f"{prefix}/active_run.json"}
    keys.update(f"{root}/checkpoints/epoch_{epoch:03d}.pt" for epoch in range(1, epochs + 1))
    keys.update(f"{root}/checkpoints/{name}" for name in (
        "last_checkpoint.pt", "best_model.pt", "emergency_checkpoint.pt", "checkpoint_metadata.json",
    ))
    keys.add(f"{root}/status/step8_session_summary.json")
    keys.update(f"{root}/artifacts/{name}" for name in ARTIFACT_FILES)
    return keys


def main() -> None:
    normalize_aws_environment()
    import boto3
    from botocore.config import Config

    args = parse_args()
    if not 1 <= args.expires <= 604800:
        raise ValueError("--expires must be between 1 and 604800 seconds")
    prefix = args.s3_prefix.strip("/")
    kwargs = {"config": Config(signature_version="s3v4", s3={"addressing_style": "virtual"})}
    if args.region:
        kwargs["region_name"] = args.region
    client = boto3.client("s3", **kwargs)
    run_id, active = resolve_run(client, args.bucket, prefix, args.run_id)
    keys = run_keys(prefix, run_id, args.epochs)
    uploads = {key: client.generate_presigned_url(
        "put_object", Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=args.expires,
    ) for key in sorted(keys)}
    downloads = {key: client.generate_presigned_url(
        "get_object", Params={"Bucket": args.bucket, "Key": key}, ExpiresIn=args.expires,
    ) for key in sorted(keys)}
    payload = {
        "bucket": args.bucket, "s3_prefix": prefix, "run_id": run_id,
        "run_root": f"{prefix}/gc-lstm-ghostnet/{run_id}",
        "active_key": f"{prefix}/active_run.json",
        "aws_region": client.meta.region_name,
        "expires_at_utc": (datetime.now(timezone.utc) + timedelta(seconds=args.expires)).isoformat(),
        "resume_required": bool(active and (
            int(active.get("completed_epoch", 0)) > 0 or int(active.get("progress_cursor", 0)) > 0
            or int(active.get("consumed_train_sequences", 0)) > 0
        )),
        "contract_version": "exhaustive-label-v1",
        "uploads": uploads, "downloads": downloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Generated presigned GhostNet config for {run_id}: {len(keys)} object keys")


if __name__ == "__main__":
    main()
