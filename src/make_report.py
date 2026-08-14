from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .viz import generate_all


def main() -> None:
    parser=argparse.ArgumentParser(description="Regenerate figures and CSVs from existing artifacts only")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--s3-bucket")
    parser.add_argument("--s3-prefix")
    parser.add_argument("--aws-region")
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    if bool(args.artifact_dir) == bool(args.s3_bucket):
        raise ValueError("Provide exactly one of --artifact-dir or --s3-bucket")
    temporary = None
    artifact_dir = args.artifact_dir
    if args.s3_bucket:
        import boto3

        temporary = tempfile.TemporaryDirectory(prefix="gc_lstm_report_")
        artifact_dir = temporary.name
        client = boto3.client("s3", region_name=args.aws_region)
        token = None
        while True:
            request = {"Bucket": args.s3_bucket, "Prefix": (args.s3_prefix or "").strip("/")}
            if token:
                request["ContinuationToken"] = token
            response = client.list_objects_v2(**request)
            for item in response.get("Contents", []):
                key = item["Key"]
                relative = key[len(request["Prefix"]):].lstrip("/")
                if not relative:
                    continue
                target = Path(artifact_dir) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(args.s3_bucket, key, str(target))
            if not response.get("IsTruncated"):
                break
            token = response["NextContinuationToken"]
    try:
        print(json.dumps(generate_all(artifact_dir,args.output_dir),indent=2))
    finally:
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
