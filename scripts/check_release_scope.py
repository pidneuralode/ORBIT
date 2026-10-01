"""Finite release-scope checks; reports categories/paths, never matched contents."""

from __future__ import annotations

import re
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Already-public project-site files are preserved byte-for-byte, outside the core export.
# A changed payload at these paths is rejected, not granted an artifact exemption.
PUBLIC_SITE_SHA256 = {
    "docs/README.md": "284dc2294a02df1e8f1597206e4e369706330cbe0478bf9fa11a2a14e073a00f",
    "docs/index.html": "f54438c36a4d2e07ab04e42d2114be9b9850df5e609f21811a213cb054b48fe2",
    "docs/static/assets/judge-rubric-panel-1.png": "2c7a36a3c2f92c6d751d682083490c4dbf77d5cb0b383006a34932c2c0b33f24",
    "docs/static/assets/orbit-performance-1.png": "ec1ef54194562fb656adb3daa7990371366e97f6d77af873e3d392da158ce885",
    "docs/static/assets/orbit-pipeline-1.png": "424fed5e6a827f591927ed519f0d6f3a3d0d88c64c70571c8fdf9e0ec397e5d3",
    "docs/static/assets/performance-compare-1.png": "cab8d5e9a971811631aadf8a6a38700e9593f557e54c26123a84bcef3fcd592a",
    "docs/static/assets/training-dynamics-1.png": "3d47a4f50766a919659fbc4dd4ec8ced6338e178cde1751e255e41a99ef2ffbe",
    "docs/static/favicon.svg": "b52df14fb9e7e9ef5e6e3ad3cf503512553df3c0abf747a052954021e3f066e4",
    "docs/static/style.css": "6704871def7158ead8338ce15d4d4ba6fa20c5152a848b6d04830e6a32209431",
}
PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "token-literal": re.compile(r"\b(?:sk-[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "credential-url": re.compile(r"https?://[^\s/:]+:[^\s/@]+@"),
    "private-location": re.compile(r"/zju_\d+|/(?:Users)/[^/]+/|/(?:root)/"),
    "private-address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
    ),
}
FORBIDDEN = {".parquet", ".csv", ".xlsx", ".pt", ".pth", ".safetensors", ".bin", ".pem", ".key"}


def scan_paths(
    root: Path, names: list[str], payloads: dict[str, bytes] | None = None
) -> list[tuple[str, str]]:
    failures = []
    for name in names:
        path = root / name
        if (
            name.split("/")[0] in {"source", "audit", "archive", ".venv"}
            or path.suffix in FORBIDDEN
        ):
            failures.append((name, "prohibited-artifact"))
        payload = payloads[name] if payloads is not None else path.read_bytes()
        if name in PUBLIC_SITE_SHA256:
            if sha256(payload).hexdigest() != PUBLIC_SITE_SHA256[name]:
                failures.append((name, "changed-project-site"))
            continue
        try:
            contents = payload.decode("utf-8")
        except UnicodeError:
            failures.append((name, "non-text-artifact"))
            continue
        # Exact invalid-URL fixture exception; no entire file (including this scanner) is exempt.
        if name == "tests/test_judge.py":
            contents = contents.replace(
                "https://" + "user:secret@example.invalid", "synthetic-invalid-url"
            )
        for category, pattern in PATTERNS.items():
            if pattern.search(contents):
                failures.append((name, category))
    return failures


def index_payloads(root: Path) -> dict[str, bytes]:
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "-z"], cwd=root, check=True, capture_output=True
    )
    names = list(dict.fromkeys(name for name in listing.stdout.decode().split("\0") if name))
    if any("\n" in name or "\r" in name for name in names):
        raise ValueError("Unsupported release filename")
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="".join(":" + name + "\n" for name in names).encode(),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    payloads = {}
    offset = 0
    for name in names:
        end = batch.index(b"\n", offset)
        header = batch[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("Release index must contain regular file blobs")
        size = int(header[2])
        payloads[name] = batch[end + 1 : end + 1 + size]
        offset = end + size + 2
    return payloads


def main() -> int:
    payloads = index_payloads(ROOT)
    failures = [
        (name, "index:" + category) for name, category in scan_paths(ROOT, list(payloads), payloads)
    ]
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    names = list(
        dict.fromkeys(
            name for name in result.stdout.decode().split("\0") if name and (ROOT / name).is_file()
        )
    )
    failures += [(name, "worktree:" + category) for name, category in scan_paths(ROOT, names)]
    for name, category in failures:
        print(f"{category}: {name}")
    print(f"Release scope: {len(failures)} findings")
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
