import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from export_release import export_release
finally:
    sys.path.pop(0)


def repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def commit(root: Path) -> None:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic",
            "-c",
            "user.email=synthetic@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )


def test_export_uses_reviewed_head_not_worktree_or_history(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "README.md").write_text("old")
    commit(root)
    (root / "README.md").write_text("approved")
    commit(root)
    (root / "README.md").write_text("uncommitted")
    (root / "audit").mkdir()
    (root / "audit/private.json").write_text(json.dumps({"internal": True}))
    destination = tmp_path / "release"
    report = export_release(root, destination)
    assert report["file_count"] == 1
    assert (destination / "README.md").read_text() == "approved"
    assert not (destination / ".git").exists() and not (destination / "audit").exists()
    assert (root / "README.md").read_text() == "uncommitted"
    with pytest.raises(ValueError):
        export_release(root, destination)


def test_rejected_head_never_creates_destination(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "README.md").write_text("sk-" + "x" * 30)
    commit(root)
    destination = tmp_path / "release"
    with pytest.raises(ValueError):
        export_release(root, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".orbit-export-*"))


def test_symlink_unapproved_paths_and_checkout_destination_rejected(tmp_path):
    for item in ["symlink", "raw.parquet", "unapproved.txt"]:
        root = repository(tmp_path / item.replace(".", "-"))
        (root / "README.md").write_text("approved")
        if item == "symlink":
            (root / "docs").mkdir()
            (root / "docs/link").symlink_to("../README.md")
        else:
            (root / item).write_text("not approved")
        commit(root)
        with pytest.raises(ValueError):
            export_release(root, tmp_path / (item + "-release"))
        with pytest.raises(ValueError):
            export_release(root, root / "release")


def test_write_failure_cleans_own_temporary_tree_and_preserves_existing(tmp_path, monkeypatch):
    root = repository(tmp_path / "repo")
    (root / "README.md").write_text("approved")
    commit(root)
    destination = tmp_path / "release"
    original_write = Path.write_bytes

    def fail_export_write(path, data):
        if path.parent.name.startswith(".orbit-export-"):
            raise OSError("synthetic disk failure")
        return original_write(path, data)

    monkeypatch.setattr(Path, "write_bytes", fail_export_write)
    with pytest.raises(OSError):
        export_release(root, destination)
    assert not destination.exists() and not list(tmp_path.glob(".orbit-export-*"))


def test_symlink_parent_cannot_redirect_export_into_checkout(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "README.md").write_text("approved")
    commit(root)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        export_release(root, alias / "release")


def test_destination_symlink_race_cannot_overwrite_external_files(tmp_path, monkeypatch):
    import export_release as module

    root = repository(tmp_path / "repo")
    (root / "README.md").write_text("approved")
    commit(root)
    external = tmp_path / "external"
    external.mkdir()
    (external / "README.md").write_text("must survive")
    destination = tmp_path / "release"
    original = module.publish_directory

    def race(source, target):
        target.symlink_to(external, target_is_directory=True)
        original(source, target)

    monkeypatch.setattr(module, "publish_directory", race)
    with pytest.raises(OSError):
        export_release(root, destination)
    assert (external / "README.md").read_text() == "must survive"
    assert destination.is_symlink()
    assert not list(tmp_path.glob(".orbit-export-*"))


def test_export_preserves_repo_site_but_excludes_it_from_core(tmp_path, monkeypatch):
    import export_release as module

    root = repository(tmp_path / "repo")
    (root / "README.md").write_text("approved core")
    name = "docs/static/assets/orbit-pipeline-1.png"
    original = b"\x89PNG\r\n\x1a\nsynthetic-public-site"
    site = root / name
    site.parent.mkdir(parents=True)
    site.write_bytes(original)
    monkeypatch.setitem(module.PUBLIC_SITE_SHA256, name, sha256(original).hexdigest())
    commit(root)
    destination = tmp_path / "release"
    assert export_release(root, destination)["file_count"] == 1
    assert site.read_bytes() == original
    assert not (destination / name).exists()
    site.write_bytes(original + b"changed")
    commit(root)
    with pytest.raises(ValueError):
        export_release(root, tmp_path / "changed-release")
    assert not (tmp_path / "changed-release").exists()
