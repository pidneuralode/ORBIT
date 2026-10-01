import importlib.util
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_scope", ROOT / "scripts/check_release_scope.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_scanner_is_not_exempt_and_untracked_scope_can_be_checked(tmp_path):
    folder = tmp_path / "scripts"
    folder.mkdir()
    path = folder / "check_release_scope.py"
    path.write_text('key = "sk-' + "x" * 30 + '"')
    assert module.scan_paths(tmp_path, ["scripts/check_release_scope.py"]) == [
        ("scripts/check_release_scope.py", "token-literal")
    ]


def test_research_image_bytes_are_excluded(tmp_path):
    folder = tmp_path / "docs/figures"
    folder.mkdir(parents=True)
    path = folder / "figure.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic")
    assert module.scan_paths(tmp_path, ["docs/figures/figure.png"]) == [
        ("docs/figures/figure.png", "non-text-artifact")
    ]


def test_index_scans_staged_blob_not_cleaned_worktree(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = tmp_path / "module.py"
    path.write_text('key = "sk-' + "x" * 30 + '"')
    subprocess.run(["git", "add", "module.py"], cwd=tmp_path, check=True)
    path.write_text("key = None")
    payloads = module.index_payloads(tmp_path)
    assert module.scan_paths(tmp_path, list(payloads), payloads) == [("module.py", "token-literal")]
    assert module.scan_paths(tmp_path, ["module.py"]) == []


def test_preserved_site_requires_exact_payload_in_index_and_worktree(tmp_path, monkeypatch):
    name = "docs/static/assets/orbit-pipeline-1.png"
    original = b"\x89PNG\r\n\x1a\nsynthetic-public-site"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes(original)
    monkeypatch.setitem(module.PUBLIC_SITE_SHA256, name, sha256(original).hexdigest())
    assert module.scan_paths(tmp_path, [name]) == []
    path.write_bytes(original + b"changed")
    assert module.scan_paths(tmp_path, [name]) == [(name, "changed-project-site")]
    assert module.scan_paths(tmp_path, [name], {name: original}) == []
    assert module.scan_paths(tmp_path, [name], {name: b"changed"}) == [
        (name, "changed-project-site")
    ]
