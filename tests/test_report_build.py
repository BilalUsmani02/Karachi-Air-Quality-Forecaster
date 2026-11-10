"""Tests for docs/report/build.py — the Markdown -> Word report builder."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("report_build", ROOT / "docs" / "report" / "build.py")
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)


def test_word_count_skips_front_matter_code_images_and_table_rules():
    md = (
        "---\ntitle: Ignored words here\n---\n"
        "One two three.\n"
        "```python\nnot = counted\n```\n"
        "![caption words](x.png)\n"
        "|---|---|\n"
        "| cell one | cell two |\n"
    )
    assert build.word_count(md) == 3 + 4  # prose + table cell words ("|" is not a word)


def test_pandoc_command_targets_docx_with_toc_numbering_and_repo_resources(tmp_path):
    src = tmp_path / "x_report.md"
    cmd = build.pandoc_command(src, tmp_path / "x_report.docx")
    assert cmd[0] == "pandoc" and "--to=docx" in cmd and "--toc" in cmd and "--number-sections" in cmd
    assert any(arg.startswith("--resource-path=") and str(build.ROOT) in arg for arg in cmd)


def test_build_raises_a_clear_error_for_a_missing_report():
    with pytest.raises(FileNotFoundError, match="does not exist yet"):
        build.build("no_such_report")
