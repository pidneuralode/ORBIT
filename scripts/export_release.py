"""Export scanned HEAD source without copying history, local data or audit files."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

from check_release_scope import PUBLIC_SITE_SHA256, scan_paths

ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "constraints-dev-tested.txt",
}
ROOT_DIRS = {"src", "tests", "docs", "scripts", ".github"}
EXAMPLE_FILES = {
    name + ".json"
    for name in (
        "fixed-responses",
        "generation-context",
        "prepared-responses",
        "prepared-train",
        "prepared-val",
        "rag-corpus",
        "rag-query",
        "recorded-generation",
        "recorded-judge",
        "synthetic_case",
    )
}
CONFIG_FILES = {
    "configs/offline.json",
    "configs/training.json",
    "configs/generation.json",
    "configs/offline-generation.yaml",
    "configs/generate.yaml",
    "configs/train.yaml",
} | {
    "configs/experiments/" + name + ".json"
    for name in (
        "adaptive_all",
        "baseline",
        "curriculum",
        "curriculum_admission",
        "ordered_batch",
        "rag_consensus",
        "rag_hard",
        "rag_no_hard",
        "rag_no_rag",
        "rag_standard",
        "rag_supp_no_rag",
        "stochastic_review",
    )
}
EXCLUDED = {
    ".git",
    ".venv",
    "audit",
    "archive",
    "source",
    "__pycache__",
    "outputs",
    "checkpoints",
    "datasets",
}


def head_payloads(root: Path) -> tuple[str, dict[str, bytes], dict[str, int]]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    listing = subprocess.run(
        ["git", "ls-tree", "-rz", "--full-tree", revision],
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    entries = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, name_bytes = entry.split(b"\t", 1)
        mode, kind, digest = metadata.decode().split()
        name = name_bytes.decode("utf-8")
        parts = PurePosixPath(name).parts
        if (
            not parts
            or any(part in EXCLUDED or part in {".", ".."} for part in parts)
            or PurePosixPath(name).is_absolute()
        ):
            raise ValueError("Prohibited release path")
        allowed = (
            name in ROOT_FILES
            or parts[0] in ROOT_DIRS
            or name.startswith("vendor/verl/")
            or name in CONFIG_FILES
            or (parts[0] == "examples" and len(parts) == 2 and parts[1] in EXAMPLE_FILES)
        )
        if not allowed or kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("Release tree contains unapproved path or file mode")
        entries.append((name, digest, int(mode, 8)))
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        input="".join(digest + "\n" for _, digest, _ in entries).encode(),
        cwd=root,
        check=True,
        capture_output=True,
    ).stdout
    payloads = {}
    modes = {}
    offset = 0
    for name, _, mode in entries:
        end = batch.index(b"\n", offset)
        header = batch[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError("Invalid source blob")
        size = int(header[2])
        contents = batch[end + 1 : end + 1 + size]
        if name in PUBLIC_SITE_SHA256:
            if scan_paths(root, [name], {name: contents}):
                raise ValueError("Existing project-site payload changed")
        else:
            payloads[name] = contents
            modes[name] = mode
        offset = end + size + 2
    return revision, payloads, modes


def publish_directory(source: Path, destination: Path) -> None:
    """Atomic directory rename with kernel no-replace semantics; no racy fallback."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        function = library.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        function.restype = ctypes.c_int
        # RENAME_EXCL from the macOS SDK sys/stdio.h.
        result = function(os.fsencode(source), os.fsencode(destination), 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        function = library.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        function.restype = ctypes.c_int
        # AT_FDCWD and RENAME_NOREPLACE from Linux's public rename interface.
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        raise RuntimeError(
            "Atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))


def export_release(root: Path, destination: Path) -> dict[str, object]:
    root = root.resolve()
    destination = destination.parent.resolve() / destination.name
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise ValueError("Export requires a new destination with an existing parent")
    if destination == root or root in destination.parents:
        raise ValueError("Export destination must be outside the checkout")
    revision, payloads, modes = head_payloads(root)
    failures = scan_paths(root, list(payloads), payloads)
    if failures:
        categories = sorted({category for _, category in failures})
        raise ValueError("Release scan rejected categories: " + ", ".join(categories))
    temporary = Path(tempfile.mkdtemp(prefix=".orbit-export-", dir=destination.parent))
    try:
        for name, contents in payloads.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            path.chmod(modes[name] & 0o777)
        if scan_paths(temporary, list(payloads)):
            raise ValueError("Exported files failed verification")
        publish_directory(temporary, destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return {
        "revision": revision,
        "file_count": len(payloads),
        "history_included": False,
        "published": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(
            json.dumps(
                export_release(Path(__file__).resolve().parents[1], args.destination), indent=2
            )
        )
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError):
        print(
            "Export failed; verify approved HEAD scope and a new destination. No publication occurred."
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
