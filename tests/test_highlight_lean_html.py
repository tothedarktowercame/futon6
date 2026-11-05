import importlib.util
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "highlight-lean-html.py"


def load_module():
    spec = importlib.util.spec_from_file_location("highlight_lean_html", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_highlight_escapes_source_and_marks_lean_tokens():
    rendered = load_module().highlight("theorem demo : 1 < 2 := by\n  omega")
    assert 'class="lean-decl"' in rendered
    assert 'class="lean-num"' in rendered
    assert "&lt;" in rendered
    assert rendered.startswith('<pre class="ltx_verbatim ltx_font_typewriter lean-highlight">')


def test_cli_replaces_matching_latexml_block_and_installs_css(tmp_path):
    markdown = tmp_path / "proof.md"
    output = tmp_path / "proof.html"
    markdown.write_text("```lean\ntheorem demo : True := by trivial\n```\n")
    output.write_text(
        '<html><head></head><body><pre class="ltx_verbatim ltx_font_typewriter">'
        "theorem demo : True := by trivial\n</pre></body></html>"
    )

    result = subprocess.run(
        [str(SCRIPT), str(markdown), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    rendered = output.read_text()
    assert "highlighted 1 Lean block(s)" in result.stdout
    assert 'id="lean-proofread-highlighting"' in rendered
    assert "lean-highlight" in rendered
    assert 'class="lean-decl"' in rendered
