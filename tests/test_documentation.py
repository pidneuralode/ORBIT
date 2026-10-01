import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_local_links_resolve_and_research_figures_are_absent():
    readme = (ROOT / "README.md").read_text()
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme):
        if not target.startswith(("https://", "http://", "#")):
            assert (ROOT / target.split("#")[0]).is_file(), target
    assert not (ROOT / "docs/figures").exists()
    assert not re.search(r"!\[[^\]]*\]\(", readme)
