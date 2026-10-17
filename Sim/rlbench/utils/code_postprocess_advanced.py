
"""
LLM-generated Python code post-processor
---------------------------------------

This script pipelines multiple tools to clean and validate Python code that
came from an LLM (often embedded in Markdown).

Included steps (graceful fallbacks if a tool is missing):
1) Extract code blocks from Markdown / raw text (regex + optional Pygments).
2) Language detect per block (label or Pygments) and keep Python.
3) Optional prompt cleanup (remove ">>>"/"..." from pycon-style snippets).
4) Syntax check via Python AST.
5) Autofix: autoflake (unused imports/vars) → black (formatting).
6) libcst pass for simple safety rewrites (e.g., bare `except:` → `except Exception:`).
7) Static analysis: pyflakes and/or pylint (does not execute code).
8) Emit final code and a human-readable report.

Install (pick what you need):
    pip install black pygments autoflake libcst pyflakes pylint markdown

Usage examples:
    # From a Markdown file with mixed content
    python postprocess_llm_code.py -i llm_output.md -o cleaned.py --report

    # From stdin → stdout
    cat raw.txt | python postprocess_llm_code.py > cleaned.py

Notes:
- tree-sitter (Python) can be integrated if you have a compiled language lib,
  but many environments don’t. This script prioritizes stdlib/Python-native
  libs and provides optional stubs for tree-sitter checks when available.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple
from utils.code_postprocess import add_return_statement


try:
    import black
except Exception:
    black = None

try:
    import autoflake
except Exception:
    autoflake = None

try:
    import libcst as cst
except Exception:
    cst = None

try:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
except Exception:
    guess_lexer = None
    ClassNotFound = Exception

try:
    import markdown
except Exception:
    markdown = None


try:
    from tree_sitter import Language, Parser

    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False


PYTHON_LANG_LABELS = {
    "py",
    "python",
    "python3",
    "py3",
    "pycon",
}


@dataclass
class CodeBlock:
    language: Optional[str]
    code: str
    fence_label: Optional[str] = None


@dataclass
class Diagnostics:
    syntax_ok: bool
    syntax_error: Optional[str]
    pyflakes: Optional[str]
    pylint: Optional[str]
    steps_run: List[str]


@dataclass
class Config:
    strip_prompts: bool = True
    join_blocks_with: str = "\n\n"
    prefer_labeled_language: bool = True
    run_autoflake: bool = True
    run_black: bool = True
    run_libcst: bool = True
    run_pyflakes: bool = True
    run_pylint: bool = False
    pylint_max_line_length: int = 100
    add_header_comment: bool = True






FENCE_RE = re.compile(
    r"```(?P<label>[a-zA-Z0-9_+\-]*)\n(?P<code>[\s\S]*?)```",
    re.MULTILINE,
)


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """Extract triple-fenced code blocks. If none found, treat whole text as one block."""
    blocks: List[CodeBlock] = []
    for m in FENCE_RE.finditer(text):
        label = m.group("label").strip() or None
        code = m.group("code")
        blocks.append(CodeBlock(language=label, code=code, fence_label=label))

    if not blocks:

        blocks = [CodeBlock(language=None, code=text, fence_label=None)]
    return blocks


def normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    L = label.lower().strip()
    if L in PYTHON_LANG_LABELS:
        return "python"
    return L


def detect_language(block: CodeBlock) -> str:
    """Return a language guess; prefer explicit labels when configured."""
    label = normalize_label(block.language)
    if label in ("python", "pycon"):
        return "python"


    if guess_lexer is None:
        return label or "text"
    try:
        lx = guess_lexer(block.code)
        name = lx.name.lower()
        if "python" in name:
            return "python"
        return name or (label or "text")
    except ClassNotFound:
        return label or "text"






PYCON_PROMPT_RE = re.compile(r"^(>>> |\.\.\. )", re.MULTILINE)


def strip_pycon_prompts(code: str) -> str:

    return PYCON_PROMPT_RE.sub("", code)


MD_INLINE_CODE_LINE_RE = re.compile(r"^`{1,3}.*?`{1,3}$")


def drop_inline_code_lines(code: str) -> str:

    lines = [ln for ln in code.splitlines() if not MD_INLINE_CODE_LINE_RE.match(ln.strip())]
    return "\n".join(lines)







def python_syntax_ok(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"


def longest_parsable_prefix(code: str) -> Tuple[str, int]:
    """Return the longest prefix (by lines) that parses as valid Python.
    If nothing parses, returns empty string and 0.
    """
    lines = code.splitlines()
    best_idx = 0
    for i in range(1, len(lines) + 1):
        chunk = """
""".join(lines[:i])
        ok, _ = python_syntax_ok(chunk)
        if ok:
            best_idx = i
    if best_idx == 0:
        return "", 0
    return """
""".join(lines[:best_idx]).rstrip() + """
""", best_idx


def salvage_incomplete_code(code: str, max_trim_ratio: float = 0.35) -> Tuple[str, Optional[str]]:
    """Heuristically salvage code when LLM output is truncated or has dangling blocks.
    Strategy: keep the longest parsable prefix by lines. If we had to trim more than
    `max_trim_ratio` of lines, we still return the prefix but note it in diagnostics.
    """
    prefix, kept = longest_parsable_prefix(code)
    total = len(code.splitlines())
    if kept == 0:
        return code, "salvage_failed_no_parsable_prefix"
    trimmed = total - kept
    if trimmed > 0:
        note = f"salvaged_prefix_lines={kept}/{total} (trimmed={trimmed})"
    else:
        note = None
    return prefix, note


def _run_module(module: str, args: List[str]) -> Tuple[int, str, str]:
    """Run a tool as a Python module (portable on Windows)."""
    proc = subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def run_pyflakes_on_code(code: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        rc, out, err = _run_module("pyflakes", [tmp_path])
        return (out + err).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def run_pylint_on_code(code: str, max_line_length: int = 100) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        rc, out, err = _run_module(
            "pylint",
            [
                "--disable=R,C",
                f"--max-line-length={max_line_length}",
                tmp_path,
            ],
        )
        return (out + err).strip()
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def run_autoflake(code: str) -> str:
    if autoflake is None:
        return code
    try:
        return autoflake.fix_code(
            code,
            remove_all_unused_imports=True,
            remove_unused_variables=True,
            expand_star_imports=False,
        )
    except Exception:
        return code


def run_black(code: str) -> str:
    if black is None:
        return code
    try:
        mode = black.Mode()
        return black.format_str(code, mode=mode)
    except Exception:
        return code

    return black.format_str(code, mode=mode)



if cst is not None:

    class FixBareExcept(cst.CSTTransformer):
        def leave_ExceptHandler(
            self, original_node: "cst.ExceptHandler", updated_node: "cst.ExceptHandler"
        ) -> "cst.ExceptHandler":
            if updated_node.type is None:
                return updated_node.with_changes(type=cst.Name("Exception"))
            return updated_node

    def libcst_tweaks(code: str) -> str:
        try:
            mod = cst.parse_module(code)
            new_mod = mod.visit(FixBareExcept())
            return new_mod.code
        except Exception:
            return code

else:

    def libcst_tweaks(code: str) -> str:
        return code






def treesitter_quick_check_python(code: str) -> bool:
    if not _TS_AVAILABLE:
        return True
    try:

        lib = os.environ.get("TS_LANG")
        if not lib or not os.path.exists(lib):
            return True
        lang = Language(lib, "python")
        parser = Parser()
        parser.set_language(lang)
        tree = parser.parse(code.encode("utf-8"))
        return tree.root_node.has_error is False
    except Exception:
        return True







def postprocess_text(text: str, cfg: Config) -> Tuple[str, Diagnostics]:
    steps: List[str] = []


    raw_blocks = extract_code_blocks(text)
    steps.append(f"extracted_blocks={len(raw_blocks)}")


    py_blocks: List[CodeBlock] = []
    for b in raw_blocks:
        lang = detect_language(b)
        if lang == "python":
            code = b.code
            if cfg.strip_prompts:
                code = strip_pycon_prompts(code)
            code = drop_inline_code_lines(code)
            py_blocks.append(CodeBlock(language="python", code=code, fence_label=b.fence_label))

    if not py_blocks:
        only = raw_blocks[0]
        code = only.code
    else:
        code = cfg.join_blocks_with.join([b.code.strip("""
""") for b in py_blocks])


    ok, err = python_syntax_ok(code)
    steps.append(f"syntax_pre={'ok' if ok else 'fail'}")


    if not ok:
        salvaged, note = salvage_incomplete_code(code)
        if note:
            steps.append(note)
        ok2, _ = python_syntax_ok(salvaged)
        if ok2:
            code = salvaged
            steps.append("salvage_applied")
        else:
            steps.append("salvage_not_effective")


    if cfg.run_autoflake:
        code = run_autoflake(code)
        steps.append("autoflake")


    if cfg.run_black:
        code = run_black(code)
        steps.append("black")


    if cfg.run_libcst:
        code = libcst_tweaks(code)
        steps.append("libcst")
        if cfg.run_black:
            code = run_black(code)


    if not treesitter_quick_check_python(code):
        steps.append("treesitter=error")
    else:
        steps.append("treesitter=ok_or_skipped")


    syntax_ok, syntax_error = python_syntax_ok(code)


    pyflakes_out = run_pyflakes_on_code(code) if cfg.run_pyflakes else None
    pylint_out = run_pylint_on_code(code, cfg.pylint_max_line_length) if cfg.run_pylint else None

    diag = Diagnostics(
        syntax_ok=syntax_ok,
        syntax_error=syntax_error,
        pyflakes=pyflakes_out,
        pylint=pylint_out,
        steps_run=steps,
    )
    return code, diag

def fix_if_done_return(code: str) -> str:
    """
    Find lines with 'if done:' and check the next non-empty line.
    If that line does not start with 'return' (ignoring leading spaces),
    replace the 'if done:' line with 'if done:    return'.
    """
    lines = code.splitlines()
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("if done:"):

            indent = line[: len(line) - len(stripped)]

            if i + 1 < len(lines):
                nxt = lines[i + 1].lstrip()
                if not nxt.startswith("return"):

                    new_lines.append(f"{indent}if done:    return")
                    i += 1
                    continue
            else:

                new_lines.append(f"{indent}if done:    return")
                i += 1
                continue
        new_lines.append(line)
        i += 1
    return "\n".join(new_lines)






def postprocess_advanced(file_name, code = None, no_autoflake = False, no_black = False, no_libcst = False, no_pyflakes = False, pylint = True, max_line_length = 100, keep_prompts = False, report = False):
    text: str
    if code is None:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = code

    cfg = Config(
        strip_prompts=not keep_prompts,
        run_autoflake=not no_autoflake,
        run_black=not no_black,
        run_libcst=not no_libcst,
        run_pyflakes=not no_pyflakes,
        run_pylint=bool(pylint),
        pylint_max_line_length=int(max_line_length),
    )


    text = fix_if_done_return(code=text)
    code, diag = postprocess_text(text, cfg)
    code = add_return_statement(code)

    if file_name:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        sys.stdout.write(code)

    if report:
        sys.stderr.write("\n=== Diagnostics ===\n")
        sys.stderr.write(f"Steps: {', '.join(diag.steps_run)}\n")
        sys.stderr.write(f"Syntax OK: {diag.syntax_ok}\n")
        if diag.syntax_error:
            sys.stderr.write(diag.syntax_error + "\n")
        if diag.pyflakes is not None:
            sys.stderr.write("\n[pyflakes]\n" + (diag.pyflakes or "(no issues)") + "\n")
        if diag.pylint is not None:
            sys.stderr.write("\n[pylint]\n" + (diag.pylint or "(no issues)") + "\n")

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Python code post-processor")
    p.add_argument("-i", "--input", type=str, help="Input file (Markdown or text). If omitted, read stdin.")
    p.add_argument("-o", "--output", type=str, help="Output .py file. If omitted, print to stdout.")
    p.add_argument("--no-autoflake", action="store_true", help="Disable autoflake pass.")
    p.add_argument("--no-black", action="store_true", help="Disable black formatting.")
    p.add_argument("--no-libcst", action="store_true", help="Disable libcst tweaks.")
    p.add_argument("--no-pyflakes", action="store_true", help="Disable pyflakes check.")
    p.add_argument("--pylint", action="store_true", help="Enable pylint check (off by default).")
    p.add_argument("--max-line-length", type=int, default=100, help="Pylint max line length.")
    p.add_argument("--keep-prompts", action="store_true", help="Keep pycon prompts (>>> / ...).")
    p.add_argument("--report", action="store_true", help="Print diagnostics to stderr.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    text: str
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    cfg = Config(
        strip_prompts=not args.keep_prompts,
        run_autoflake=not args.no_autoflake,
        run_black=not args.no_black,
        run_libcst=not args.no_libcst,
        run_pyflakes=not args.no_pyflakes,
        run_pylint=bool(args.pylint),
        pylint_max_line_length=int(args.max_line_length),
    )

    code, diag = postprocess_text(text, cfg)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        sys.stdout.write(code)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
