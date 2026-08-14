"""Decide whether the next CPU-only GhostNet Kaggle session should be launched."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from .kaggle_http import get_kernel_status
except ImportError:
    from kaggle_http import get_kernel_status

ROOT = Path(__file__).resolve().parents[1]


def clean_environment() -> None:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_SECRET", "KAGGLE_KERNEL", "AWS_ACCESS_KEY_ID",
                 "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION", "AWS_DEFAULT_REGION", "S3_BUCKET", "S3_PREFIX"):
        value = os.environ.get(name)
        if value is not None:
            os.environ[name] = value.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "").strip()


def timestamp(value: str | None) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() if value else None
    except ValueError:
        return None


def normalize_status(value: str) -> str:
    text = value.casefold()
    for status, tokens in (("running", ("running",)), ("queued", ("queued", "pending")),
                           ("complete", ("complete",)), ("cancelled", ("cancelled", "canceled")),
                           ("error", ("error", "failed", "failure"))):
        if any(token in text for token in tokens):
            return status
    return "unknown"


def progress(active: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(active.get("completed_epoch", 0)), int(active.get("active_epoch", 1)),
            int(active.get("consumed_train_sequences", 0)))


@dataclass(frozen=True)
class Decision:
    should_push: bool
    reason: str
    completed_epoch: int
    kernel_status: str
    session_attempts: int
    stagnant_restarts: int


def decide_next_session(active: Mapping[str, Any] | None, state: Mapping[str, Any] | None,
                        kernel_status: str, config: Mapping[str, Any], now: float, force: bool = False) -> Decision:
    active, state = dict(active or {}), dict(state or {})
    completed = int(active.get("completed_epoch", 0)); status = str(active.get("status", "missing")).casefold()
    attempts = int(state.get("session_attempts", 0)); stagnant = int(state.get("stagnant_restarts", 0))
    result = lambda push, reason: Decision(push, reason, completed, kernel_status, attempts, stagnant)
    if force: return result(True, "manual force")
    if status == "completed": return result(False, "training and final artifacts are complete")
    if kernel_status in {"running", "queued"}: return result(False, f"Kaggle kernel is {kernel_status}")
    if status == "running" and kernel_status == "unknown":
        heartbeat = timestamp(active.get("updated_at")); stale = float(config["running_heartbeat_stale_hours"]) * 3600
        if heartbeat is None or now - heartbeat < stale:
            return result(False, "S3 heartbeat is not stale")
    last_push = timestamp(state.get("last_push_at"))
    if last_push is not None and now - last_push < int(config["recent_push_guard_minutes"]) * 60:
        return result(False, "recent push guard is active")
    if attempts >= int(config["maximum_session_attempts"]): return result(False, "maximum session attempts reached")
    if stagnant >= int(config["maximum_stagnant_restarts"]): return result(False, "maximum stagnant restarts reached")
    if status == "paused": reason = "previous session stopped safely and is resumable"
    elif not active: reason = "no active S3 run exists"
    else: reason = f"run is incomplete and Kaggle is {kernel_status}"
    return result(True, reason)


class S3State:
    def __init__(self) -> None:
        import boto3
        self.bucket = os.environ["S3_BUCKET"].strip(); self.prefix = os.environ["S3_PREFIX"].strip().strip("/")
        self.client = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))

    def read(self, name: str) -> dict[str, Any] | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{name}")["Body"].read()
            return json.loads(body.decode("utf-8"))
        except Exception as exc:
            response = getattr(exc, "response", None) or {}; code = response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "NotFound", "404"}: return None
            raise

    def write(self, name: str, payload: Mapping[str, Any]) -> None:
        self.client.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{name}",
                               Body=(json.dumps(dict(payload), indent=2) + "\n").encode())


def main() -> None:
    clean_environment()
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", default=ROOT / "configs" / "orchestration.json")
    sub = parser.add_subparsers(dest="command", required=True)
    decide = sub.add_parser("decide"); decide.add_argument("--kernel", required=True); decide.add_argument("--github-output"); decide.add_argument("--force", action="store_true")
    record = sub.add_parser("record-push"); record.add_argument("--reason", required=True)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text()); store = S3State()
    if args.command == "decide":
        active, state = store.read("active_run.json"), store.read("orchestration_state.json")
        try: kernel = normalize_status(get_kernel_status(args.kernel))
        except Exception: kernel = "unknown"
        decision = decide_next_session(active, state, kernel, config, time.time(), args.force)
        if args.github_output:
            with Path(args.github_output).open("a", encoding="utf-8") as handle:
                for key, value in (("should_push", str(decision.should_push).lower()), ("reason", decision.reason),
                                   ("completed_epoch", decision.completed_epoch), ("kernel_status", decision.kernel_status)):
                    handle.write(f"{key}={value}\n")
        print(json.dumps(asdict(decision), indent=2)); return
    previous, active = store.read("orchestration_state.json") or {}, store.read("active_run.json") or {}
    current_progress = progress(active); previous_progress = tuple(previous.get("last_observed_progress", [-1, -1, -1]))
    stagnant = 0 if current_progress > previous_progress else int(previous.get("stagnant_restarts", 0)) + 1
    state = {"last_push_at": datetime.now(timezone.utc).isoformat(), "last_push_reason": args.reason,
             "last_observed_progress": list(current_progress), "session_attempts": int(previous.get("session_attempts", 0)) + 1,
             "stagnant_restarts": stagnant}
    store.write("orchestration_state.json", state); print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
