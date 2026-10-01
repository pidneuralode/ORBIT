import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_resolve_and_figures_use_public_project_assets():
    readme = (ROOT / "README.md").read_text()
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme):
        if not target.startswith(("https://", "http://", "#")):
            assert (ROOT / target.split("#")[0]).is_file(), target
    assert not (ROOT / "docs/figures").exists()
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        assert target.startswith(
            "https://raw.githubusercontent.com/pidneuralode/ORBIT/main/docs/static/assets/"
        ), target
