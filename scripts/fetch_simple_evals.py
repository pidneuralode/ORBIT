"""Fetch pinned official evaluation code; never download data or run an evaluation."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPOSITORY = "https://github.com/openai/simple-evals.git"
REVISION = "652c89d0ca9df547706735883097e9537d40dc47"


def fetch(destination: Path) -> str:
    if destination.exists():
        raise FileExistsError("Choose a new destination for the upstream checkout")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-c", "credential.helper=", "clone", "--no-checkout", REPOSITORY, str(destination)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "checkout", "--detach", REVISION],
        cwd=destination,
        check=True,
        capture_output=True,
    )
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != REVISION or not (destination / "LICENSE").is_file():
        raise ValueError("Upstream revision or license verification failed")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(".external/simple-evals"))
    args = parser.parse_args()
    try:
        print("Official simple-evals checkout: " + fetch(args.destination))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(
            "Upstream checkout failed ("
            + type(error).__name__
            + "); choose a new writable destination and check Git/network access"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
