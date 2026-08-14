"""Minimal Kaggle status/push client that preserves attached dataset sources."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping

API_ROOT = "https://api.kaggle.com/v1/kernels.KernelsApiService"


def clean_token(value: str) -> str:
    return value.replace("\r", "").replace("\n", "").replace("\\r", "").replace("\\n", "").strip()


def api_token() -> str:
    value = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_API_TOKEN_SECRET")
    if not value:
        raise RuntimeError("KAGGLE_API_TOKEN secret is missing")
    return clean_token(value)


def post(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{API_ROOT}/{method}", data=json.dumps(dict(payload)).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_token()}", "Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kaggle API {method} failed with HTTP {exc.code}: {body[:1000]}") from exc


def split_kernel(kernel: str) -> tuple[str, str]:
    parts = kernel.strip().split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError("Kaggle kernel must use owner/kernel-slug format")
    return parts[0], parts[1]


def get_kernel_status(kernel: str) -> str:
    owner, slug = split_kernel(kernel)
    return str(post("GetKernelSessionStatus", {"userName": owner, "kernelSlug": slug}).get("status", "UNKNOWN"))


def metadata_bool(metadata: Mapping[str, Any], name: str, default: bool) -> bool:
    value = metadata.get(name, default)
    return value if isinstance(value, bool) else str(value).strip().casefold() in {"1", "true", "yes", "on"}


def push_kernel(bundle: Path, timeout: int) -> dict[str, Any]:
    metadata = json.loads((bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
    owner, slug = split_kernel(str(metadata["id"]))
    notebook = json.loads((bundle / str(metadata["code_file"])).read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    payload: dict[str, Any] = {
        "slug": f"{owner}/{slug}", "newTitle": str(metadata["title"]), "text": json.dumps(notebook),
        "language": str(metadata.get("language", "python")), "kernelType": str(metadata.get("kernel_type", "notebook")),
        "datasetDataSources": list(metadata.get("dataset_sources", [])),
        "kernelDataSources": list(metadata.get("kernel_sources", [])),
        "competitionDataSources": list(metadata.get("competition_sources", [])),
        "modelDataSources": list(metadata.get("model_sources", [])),
        "categoryIds": list(metadata.get("keywords", [])),
        "isPrivate": metadata_bool(metadata, "is_private", True),
        "enableGpu": metadata_bool(metadata, "enable_gpu", False),
        "enableTpu": metadata_bool(metadata, "enable_tpu", False),
        "enableInternet": metadata_bool(metadata, "enable_internet", True),
        "sessionTimeoutSeconds": int(timeout),
    }
    return post("SaveKernel", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status"); status.add_argument("--kernel", required=True)
    push = sub.add_parser("push"); push.add_argument("--path", type=Path, required=True); push.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    if args.command == "status":
        print(f'Kernel has status "{get_kernel_status(args.kernel)}"')
    else:
        response = push_kernel(args.path, args.timeout)
        print(json.dumps({key: response.get(key) for key in ("ref", "url", "versionNumber", "kernelId")}))


if __name__ == "__main__":
    main()
