"""Offline wheel smoke: install core in a fresh venv and use actual console commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="orbit-install-smoke-") as directory:
        temporary = Path(directory)
        wheels = temporary / "wheels"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-index",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(ROOT),
            ],
            check=True,
            capture_output=True,
        )
        environment_path = temporary / "environment"
        venv.EnvBuilder(with_pip=True).create(environment_path)
        scripts = environment_path / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        cli = scripts / ("orbit.exe" if os.name == "nt" else "orbit")
        wheel = next(wheels.glob("orbit_rubrics-*.whl"))
        with zipfile.ZipFile(wheel) as archive:
            for source in (
                ROOT / "LICENSE",
                ROOT / "NOTICE",
                ROOT / "docs/licenses/simple-evals-MIT.txt",
                ROOT / "docs/licenses/verl-Apache-2.0.txt",
            ):
                matches = [name for name in archive.namelist() if Path(name).name == source.name]
                assert len(matches) == 1 and archive.read(matches[0]) == source.read_bytes()
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "ORBIT_CONFIG", "ORBIT_VERL_ROOT", "ORBIT_REPO_ROOT"):
            environment.pop(key, None)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=True,
            capture_output=True,
            env=environment,
        )
        check = subprocess.run(
            [
                str(python),
                "-c",
                "import orbit,sys; print(orbit.__file__); assert 'torch' not in sys.modules and 'openai' not in sys.modules",
            ],
            cwd=temporary,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert str(ROOT / "src") not in check.stdout
        for command, input_name in [
            ("score", "synthetic_case"),
            ("generate", "generation-context"),
            ("evaluate", "fixed-responses"),
            ("rag", "rag-query"),
        ]:
            argv = [
                str(cli),
                command,
                "--config",
                str(ROOT / "configs/offline.json"),
                "--input",
                str(ROOT / "examples" / (input_name + ".json")),
            ]
            if command == "rag":
                argv += ["--corpus", str(ROOT / "examples/rag-corpus.json")]
            else:
                recorded = "recorded-generation" if command == "generate" else "recorded-judge"
                argv += ["--recorded", str(ROOT / "examples" / (recorded + ".json"))]
            result = subprocess.run(
                argv, cwd=temporary, env=environment, check=True, capture_output=True, text=True
            )
            json.loads(result.stdout)
        subprocess.run(
            [str(cli), "--help"], cwd=temporary, env=environment, check=True, capture_output=True
        )
        workflow = temporary / "workflow.json"
        workflow.write_text(
            json.dumps(
                {
                    "workflow": {
                        "command": "generate",
                        "input": str(ROOT / "examples/generation-context.json"),
                        "recorded": str(ROOT / "examples/recorded-generation.json"),
                        "output": "generated.json",
                    }
                }
            )
        )
        subprocess.run(
            [str(cli), "run", "--config", str(workflow)],
            cwd=temporary,
            env=environment,
            check=True,
            capture_output=True,
        )
        assert json.loads((temporary / "generated.json").read_text())[0]["evaluation_criteria"]
        print(
            "Fresh core wheel: license payloads, import and six console smoke checks passed; no source import, network or model calls."
        )


if __name__ == "__main__":
    main()
