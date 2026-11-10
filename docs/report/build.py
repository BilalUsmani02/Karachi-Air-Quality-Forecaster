"""Build the course reports from Markdown into Word (.docx) with pandoc.

    uv run python docs/report/build.py            # every report
    uv run python docs/report/build.py aies       # one report

Each report lives in `docs/report/<name>/<name>_report.md` and is written to
`docs/report/<name>/<name>_report.docx`. Image paths in the Markdown are resolved from the
repository root, so a report can embed `results/...png` and `docs/report/figures/...png`
directly. Maths in `$...$` / `$$...$$` becomes native Word equations.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "docs" / "report"
REPORTS = ("aies", "pai")
WORDS_PER_PAGE = 450  # rough figure for 11-12 pt text with headings; figures add pages on top


def source_path(name: str) -> Path:
    return REPORT_DIR / name / f"{name}_report.md"


def word_count(markdown: str) -> int:
    """Words of prose, ignoring YAML front matter, code blocks, image lines and table rules."""
    text = re.sub(r"\A---\n.*?\n---\n", "", markdown, flags=re.S)
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith(("![", "|---", "|:-"))]
    return len(re.findall(r"[A-Za-z0-9µ²³%.,'-]+", "\n".join(lines)))


def pandoc_command(src: Path, out: Path) -> list[str]:
    return [
        "pandoc",
        str(src),
        "--from=markdown+tex_math_dollars+pipe_tables+implicit_figures",
        "--to=docx",
        "--toc",
        "--number-sections",
        f"--resource-path={ROOT}{';' if sys.platform == 'win32' else ':'}{src.parent}",
        "-o",
        str(out),
    ]


def build(name: str) -> Path:
    src = source_path(name)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist yet")
    if shutil.which("pandoc") is None:
        raise RuntimeError("pandoc is not installed — see https://pandoc.org/installing.html")
    out = src.with_suffix(".docx")
    subprocess.run(pandoc_command(src, out), check=True, cwd=ROOT)
    words = word_count(src.read_text(encoding="utf-8"))
    print(f"{name}: wrote {out.relative_to(ROOT)} ({words} words, ~{words / WORDS_PER_PAGE:.1f} pages of text)")
    return out


def main(argv: list[str] | None = None) -> None:
    names = (argv if argv is not None else sys.argv[1:]) or [n for n in REPORTS if source_path(n).exists()]
    for name in names:
        build(name)


if __name__ == "__main__":
    main()
