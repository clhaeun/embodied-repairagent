import re
import argparse
import importlib
import ast
import sys
import json
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
import textwrap

from headers import HEADER, RUN_TASK_INTERFACE, normalize_robot_type
from ast_policy import RunTaskExtractor, get_statement_code
from prompts.prompt_utils import SUBTASK_SIGNATURES

_LIST_PREFIX = re.compile(r"^\s*(?:\[\d+\]|\d+[.)]|[-*•])\s*")

def _decomment_and_strip_list_prefix(st: str) -> str:

    s = st.lstrip("#").strip()
    s = _LIST_PREFIX.sub("", s).strip()
    return s

def _get_slice_int(slice_node) -> int | None:

    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
        return slice_node.value
    if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
        v = slice_node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, int):
            return v.value
    return None

def _is_values_call(node: ast.AST) -> bool:

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "values"
        and len(node.args) == 0
    )

class _MaxValuesSubscriptKeyFixer(ast.NodeTransformer):
    """
    Rewrite: max(X.values())[k]  ->  max(X.values(), key=lambda p: float(p[k]))[k]
    """
    def visit_Subscript(self, node: ast.Subscript):
        self.generic_visit(node)


        if not isinstance(node.value, ast.Call):
            return node
        call = node.value


        if not (isinstance(call.func, ast.Name) and call.func.id == "max"):
            return node


        if len(call.args) != 1 or not _is_values_call(call.args[0]):
            return node


        k = _get_slice_int(node.slice)
        if k is None:
            return node


        for kw in call.keywords:
            if kw.arg == "key":
                return node


        p_arg = ast.arg(arg="p")
        p_name = ast.Name(id="p", ctx=ast.Load())
        idx = ast.Subscript(
            value=p_name,
            slice=ast.Constant(value=k),
            ctx=ast.Load(),
        )
        key_expr = ast.Call(
            func=ast.Name(id="float", ctx=ast.Load()),
            args=[idx],
            keywords=[],
        )
        lam = ast.Lambda(
            args=ast.arguments(
                posonlyargs=[],
                args=[p_arg],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),
            body=key_expr,
        )
        call.keywords.append(ast.keyword(arg="key", value=lam))
        return node

def rewrite_numpy_max_patterns_in_code(code: str) -> str:
    """
    초기 코드(전체 모듈/함수 포함)에서
    max(X.values())[k] 패턴을 AST로 찾아 key=...를 자동 주입합니다.
    """
    src = (code or "").rstrip() + "\n"
    if not src.strip():
        return code

    try:
        tree = ast.parse(src)
    except Exception:
        return code

    new_tree = _MaxValuesSubscriptKeyFixer().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree).rstrip() + "\n"
    except Exception:


        return code

def rewrite_numpy_max_patterns_in_body(body: str) -> str:
    """
    continuation body(=함수 바디 조각)에 대해 numpy max 비교 패턴을 안전하게 rewrite.
    내부적으로 def __tmp__(): 로 감싸서 parse 가능하게 만든 뒤,
    rewrite_numpy_max_patterns_in_code를 적용하고 바디만 다시 추출합니다.
    """
    body = (body or "").rstrip()
    if not body.strip():
        return body

    wrapped = "def __tmp__():\n" + _normalize_body_indent(body, indent=4) + "\n"
    try:
        rewritten = rewrite_numpy_max_patterns_in_code(wrapped)
    except Exception:
        return body


    try:
        return _extract_function_body_ast(rewritten, "__tmp__").rstrip("\n")
    except Exception:
        return body






def _strip_block(text: str, begin: str, end: str) -> str:
    out = []
    in_block = False
    for line in text.splitlines():
        if line.strip() == begin:
            in_block = True
            continue
        if line.strip() == end:
            in_block = False
            continue
        if not in_block:
            out.append(line)
    return "\n".join(out)

def _strip_known_marker_blocks(text: str) -> str:
    pairs = [
        ("[EXECUTED_STATEMENTS_BEGIN]", "[EXECUTED_STATEMENTS_END]"),
        ("[EXECUTED_STATEMENTS_START]", "[EXECUTED_STATEMENTS_END]"),
        ("[ALREADY EXECUTED", "[END OF EXECUTED CODE]"),
        ("[FAILED_STATEMENT_BEGIN]", "[FAILED_STATEMENT_END]"),

        ("[REFERENCE_BEGIN]", "[REFERENCE_END]"),
        ("[CONSTRAINTS_BEGIN]", "[CONSTRAINTS_END]"),
    ]
    for b, e in pairs:
        text = _strip_block(text, b, e)
    return text

_TRIPLE_QUOTE_LEADING = re.compile(
    r'^\s*(("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'))\s*',
    re.MULTILINE
)

def _strip_leading_triple_quote_block(text: str) -> str:
    return re.sub(_TRIPLE_QUOTE_LEADING, "", text, count=1).lstrip("\n")

def _strip_prompt_blocks_and_docstrings(text: str) -> str:
    lines = text.splitlines()
    out = []
    in_marker = False
    in_triple = False
    triple_quote = None

    begin_markers = (
        "[EXECUTED_STATEMENTS_BEGIN]",
        "[EXECUTED_STATEMENTS_START]",
        "[ALREADY EXECUTED",
        "[FAILED_STATEMENT_BEGIN]",
        "[REFERENCE CODE",
    )
    end_markers = (
        "[EXECUTED_STATEMENTS_END]",
        "[END OF EXECUTED CODE]",
        "[FAILED_STATEMENT_END]",
    )

    for ln in lines:
        s = ln.strip()
        s2 = s.lstrip("#").strip()


        if any(s2.startswith(b) for b in begin_markers):
            in_marker = True
            continue
        if any(s2.startswith(e) for e in end_markers):
            in_marker = False
            continue
        if in_marker:
            continue


        if not in_triple and (s.startswith('"""') or s.startswith("'''")):
            in_triple = True
            triple_quote = s[:3]

            if s.count(triple_quote) >= 2:
                in_triple = False
                triple_quote = None
            continue
        elif in_triple:
            if triple_quote and triple_quote in s:
                in_triple = False
                triple_quote = None
            continue


        if s.startswith("# [") and "]" in s:
            continue

        out.append(ln)

    return "\n".join(out)



def _leading_spaces(s: str) -> int:
    return len(s) - len(s.lstrip(" "))
    
TOOL_CALL_PATTERNS = [
    r"<tool_call\b.*?>.*?</tool_call>",
    r"</?tool_call\b.*?>",
    r"<\|tool_call\|>.*?<\|/tool_call\|>",
    r"<\|.*?\|>",
]
def _strip_tool_calls(s: str) -> str:

    for p in TOOL_CALL_PATTERNS:
        s = re.sub(p, "", s, flags=re.DOTALL | re.IGNORECASE)


    s = re.sub(r"(<tool_call\b.*)$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r"(<\|tool_call\|>.*)$", "", s, flags=re.IGNORECASE | re.MULTILINE)

    return s

def _pick_best_code_block(s: str, subtask_name: str | None = None) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None
    if subtask_name:
        for b in blocks:
            if re.search(rf"^\s*def\s+{re.escape(subtask_name)}\s*\(", b, flags=re.MULTILINE):
                return b.rstrip()
    return blocks[0].rstrip()


def _clean_llm_text_to_python_code(raw: str, subtask_name: str | None = None) -> str:
    s = (raw or "").rstrip()

    marker_pairs = [
        ("[INITIAL_CODE_START]", "[INITIAL_CODE_END]"),
        ("[CODE_START]", "[CODE_END]"),
        ("[SUBTASK_CODE_START]", "[SUBTASK_CODE_END]"),
    ]

    extracted = None
    for sm, em in marker_pairs:
        if sm in s and em in s:
            extracted = s[s.index(sm) + len(sm): s.index(em)].rstrip()
            break

    if extracted is not None:
        s = extracted
    else:
        picked = _pick_best_code_block(s, subtask_name)
        if picked is not None:
            s = picked

    s = re.sub(r"```(?:python)?", "", s, flags=re.IGNORECASE).replace("```", "").rstrip()
    s = _strip_tool_calls(s)
    s = s.replace("\t", "    ")

    return s.rstrip() + "\n"

def _extract_function_body_ast(code_str: str, func_name: str) -> str:
    """
    '정상 파이썬 코드'에서 func_name의 body만 추출.
    여러 개가 있으면(LLM이 run_skill을 중복 생성) body가 가장 큰(스팬이 긴) 것을 선택.
    """
    tree = ast.parse(code_str)
    lines = code_str.splitlines(keepends=True)

    best = None
    best_span = -1

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name and node.body:
            start = node.body[0].lineno - 1
            end = node.body[-1].end_lineno
            span = end - start
            if span > best_span:
                best_span = span
                best = (start, end)

    if best is None:
        raise ValueError(f"Function '{func_name}' not found")

    start, end = best
    return "".join(lines[start:end])

def _normalize_indent(body: str, target_indent: int = 4) -> str:
    lines = (body or "").splitlines()


    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return "\n"

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    candidates = [ln for ln in non_empty if not ln.lstrip().startswith("#")]
    base_pool = candidates if candidates else non_empty
    min_lead = min(lead(ln) for ln in base_pool)

    prefix = " " * target_indent
    out = []
    for ln in lines:
        if ln.strip():
            out.append(prefix + ln[min_lead:])
        else:
            out.append("")
    return "\n".join(out).rstrip() + "\n"

def _normalize_body_indent(body: str, indent: int) -> str:
    """
    body 조각의 '최소 들여쓰기'를 제거한 뒤, indent만큼 다시 들여쓰기.
    (상대 들여쓰기(블록 내부)는 유지)
    """
    lines = (body or "").splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return ""

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    candidates = [ln for ln in non_empty if not ln.lstrip().startswith("#")]
    base_pool = candidates if candidates else non_empty
    min_lead = min(lead(ln) for ln in base_pool)

    prefix = " " * indent

    out = []
    for ln in lines:
        if ln.strip():
            out.append(prefix + ln[min_lead:])
        else:
            out.append("")
    return "\n".join(out).rstrip() + "\n"

_BLOCK_OPEN_RE = re.compile(r".*:\s*(#.*)?$")

def _fix_all_indentation(body: str) -> str:
    """
    모든 indentation 문제를 한 번의 순회로 해결:
    1. 블록 없이 증가한 indent를 이전 레벨로 복구
    2. 블록 내부의 과도한 indent를 expected로 제한
    3. return 문 이후 블록 컨텍스트 완전 종료
    4. if/while/for 블록 종료 시 올바른 indent로 복귀
    5. else/elif/except/finally는 이전 if/try 블록과 같은 레벨로 정렬
    """
    lines = body.splitlines()
    if not any(ln.strip() for ln in lines):
        return body

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    out = []
    stack = []
    prev_indent = 0

    for ln in lines:
        if not ln.strip():
            out.append(ln)
            continue

        cur = lead(ln)
        stripped = ln.lstrip(" ")

        is_comment = stripped.startswith("#")
        is_return = stripped.startswith("return")
        is_else_like = stripped.startswith(("else:", "elif ", "except:", "except ", "finally:"))


        if is_else_like and stack:

            stack.pop()
            if stack:

                block_indent = stack[-1][1]
            else:

                block_indent = 0
            ln = " " * block_indent + stripped
            cur = block_indent


        elif not is_else_like:
            while stack and cur <= stack[-1][0]:
                stack.pop()


        if stack and not is_else_like:

            _, expect = stack[-1]
            if cur > expect:
                ln = " " * expect + stripped
                cur = expect
        elif not is_else_like:

            if cur > prev_indent:
                ln = " " * prev_indent + stripped
                cur = prev_indent

        out.append(ln)


        if is_return:
            stack.clear()
            prev_indent = 0


        elif (not is_comment) and _BLOCK_OPEN_RE.match(ln.rstrip()):
            block_indent = cur
            stack.append((block_indent, block_indent + 4))
            prev_indent = cur
        else:
            prev_indent = cur

    return "\n".join(out).rstrip() + "\n"

def _dedent_continuation(body: str) -> str:
    lines = (body or "").splitlines()
    if not any(ln.strip() for ln in lines):
        return ""

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    non_empty = [ln for ln in lines if ln.strip()]

    candidates = [ln for ln in non_empty if not ln.lstrip().startswith("#")]


    if not candidates:
        base = min(lead(ln) for ln in non_empty)
    else:
        base = min(lead(ln) for ln in candidates)

    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        n = lead(ln)
        cut = min(n, base)
        out.append(ln[cut:])
    return "\n".join(out).rstrip() + "\n"

"""
def _find_run_skill_node(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == "run_skill":
            return node
    return None
"""
def _find_func_node(tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    return None


def merge_continuation_into_subtask(
    current_code: str,
    func_name: str,
    cont_body: str,
    insert_before_last_return: bool = True,
    insert_after_lineno: Optional[int] = None,
) -> str:
    if not (cont_body or "").strip():
        return current_code

    try:
        tree = ast.parse(current_code)
    except Exception:
        return current_code

    fn = _find_func_node(tree, func_name)
    if fn is None:
        return current_code

    lines = current_code.splitlines(keepends=True)


    body_indent = 4
    if fn.body:
        first_body_line_idx = fn.body[0].lineno - 1
        if 0 <= first_body_line_idx < len(lines):
            body_indent = _leading_spaces(lines[first_body_line_idx]) or 4


    insert_lineno = None
    if insert_after_lineno is not None:
        insert_lineno = int(insert_after_lineno) + 2
    else:
        if insert_before_last_return:
            last_ret = None
            for st in fn.body:
                if isinstance(st, ast.Return):
                    last_ret = st
            if last_ret is not None:
                insert_lineno = last_ret.lineno

    if insert_lineno is None:
        if fn.body:
            last_stmt = fn.body[-1]
            end_ln = getattr(last_stmt, "end_lineno", last_stmt.lineno)
            insert_lineno = end_ln + 1
        else:
            insert_lineno = fn.lineno + 1

    insert_idx = max(0, min(len(lines), insert_lineno - 1))


    fn_start_0 = fn.lineno - 1
    fn_end_0 = getattr(fn, "end_lineno", fn.lineno) - 1

    insert_idx = max(fn_start_0 + 1, min(insert_idx, fn_end_0 + 1))

    cont_norm = _normalize_body_indent(cont_body, indent=body_indent).rstrip("\n")
    if not cont_norm.strip():
        return current_code


    cont_norm = cont_norm + "\n\n"

    new_lines = lines[:insert_idx] + [cont_norm] + lines[insert_idx:]
    return "".join(new_lines)


def _drop_print_lines(body: str) -> str:
    out = []
    for ln in body.splitlines():
        if ln.lstrip().startswith("print("):
            continue
        out.append(ln)
    return "\n".join(out).rstrip() + "\n"

_SKILL_CALL_RE = re.compile(r"\b(ur5_|open_ur5_ee|close_ur5_ee)\w*\s*\(")

def debug_print_next_statements(
    current_code: str,
    current_lines: list[str],
    executed_stmt_texts: list[str],
    subtask_name: str,
    k: int = 8,
    tag: str = "AFTER_MERGE",
    show_full: bool = False,
):
    """
    merge 직후(또는 루프 시작 직후) 기준으로, 다음에 실행될 statement들을 미리 출력합니다.
    """
    try:
        tree = ast.parse(current_code)
    except Exception as e:

        return

    ex = RunTaskExtractor(subtask_name)
    ex.visit(tree)

    body = ex.subtask_name_body or []
    next_index = len(executed_stmt_texts)
    remaining = body[next_index:]



    for i, stmt in enumerate(remaining[:k], start=next_index + 1):
        try:
            stmt_code = get_statement_code(stmt, current_lines).rstrip()
        except Exception as e:
            stmt_code = f"<failed to render stmt: {e}>"

        kind = "RETURN" if isinstance(stmt, ast.Return) else type(stmt).__name__
        head = stmt_code.splitlines()[0] if stmt_code else ""
        lno = getattr(stmt, "lineno", None)
        end = getattr(stmt, "end_lineno", lno)



        if show_full:


    if len(remaining) > k:



class _ListAddToNpArray(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.List) and isinstance(node.right, ast.List):
                node.left = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()),
                    args=[node.left],
                    keywords=[],
                )
                node.right = ast.Call(
                    func=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="array", ctx=ast.Load()),
                    args=[node.right],
                    keywords=[],
                )
                return ast.copy_location(node, node)
        return node

def _rewrite_list_plus_list(stmt_code: str) -> str:

    if "] + [" not in stmt_code and "]+[" not in stmt_code:
        return stmt_code

    src = textwrap.dedent(stmt_code).strip()
    if not src:
        return stmt_code

    try:
        tree = ast.parse(src)
    except Exception:
        return stmt_code

    new_tree = _ListAddToNpArray().visit(tree)
    ast.fix_missing_locations(new_tree)


    try:
        new_src = ast.unparse(new_tree)
    except Exception:
        return stmt_code

    return new_src + ("\n" if not new_src.endswith("\n") else "")


def _uncomment_skill_calls(body: str) -> str:
    """
    주석으로 된 skill 호출을 실제 코드로 변환.
    예: # move_gripper_to(env, obj_name, ...) -> move_gripper_to(env, obj_name, ...)
    """
    SKILL_PREFIXES = (
        "move_gripper_to(", "move_to_position(", "move_parallel(", "rotate_gripper(",
        "open_gripper(", "close_gripper(", "pick(", "place(", "grasp_handle(", "release_handle(",
        "open_panda(", "close_panda(",
        "open_robotiq85(", "close_robotiq85(", "pick_robotiq85(", "place_robotiq85(",
        "grasp_handle_robotiq85(", "release_handle_robotiq85(",
        "activate_vacuum(", "deactivate_vacuum(", "attach_vacuum_handle(", "detach_vacuum_handle(",
        "activate_suction(", "deactivate_suction(", "attach_suction(", "detach_suction(",
        "env.get_obj_pos(", "env.get_obj_bbox(", "env.is_obj_visible(",
    )

    lines = body.splitlines()
    result = []

    for ln in lines:
        stripped = ln.strip()


        if not stripped.startswith("#"):
            result.append(ln)
            continue


        uncommented = stripped[1:].strip()


        if any(uncommented.startswith(prefix) for prefix in SKILL_PREFIXES):

            indent = ln[:len(ln) - len(ln.lstrip())]
            result.append(indent + uncommented)

        else:


            continue

    return "\n".join(result)


def postprocess_initial_code(llm_generated: str, subtask_name: str, robot_type: str, run_task_interface: Optional[str] = None,) -> str:
    raw = (llm_generated or "").rstrip()
    cleaned = _clean_llm_text_to_python_code(raw, subtask_name=subtask_name).rstrip("\n")
    if not cleaned.strip():
        cleaned = "return True"

    body = cleaned
    try:
        t0 = ast.parse(cleaned)

        if any(isinstance(n, ast.FunctionDef) and n.name == "run_task" for n in t0.body):
            body = _extract_function_body_ast(cleaned, "run_task").rstrip("\n")


        elif any(isinstance(n, ast.FunctionDef) and n.name == subtask_name for n in t0.body):
            body = _extract_function_body_ast(cleaned, subtask_name).rstrip("\n")


    except Exception:
        body = cleaned


    body = _uncomment_skill_calls(body)





    if not body.strip():
        body = "return True"




    robot_type = normalize_robot_type(robot_type)
    imports = HEADER[robot_type].strip("\n") + "\n\n"


    if subtask_name in SUBTASK_SIGNATURES:
        signature_line, description = SUBTASK_SIGNATURES[subtask_name]
        RUN_TASK_INTERFACE = signature_line + "\n"

    elif run_task_interface and run_task_interface.lstrip().startswith("def ") and run_task_interface.rstrip().endswith(":"):

        RUN_TASK_INTERFACE = run_task_interface.rstrip() + "\n"

    else:

        RUN_TASK_INTERFACE = "def {}(env, obj_name=None, target_name=None, **kwargs):\n".format(subtask_name)
        print(f"[WARN][POSTPROCESS_INIT] Using generic signature for {subtask_name}")


    if hasattr(ast, "unparse"):
        body = rewrite_numpy_max_patterns_in_code(body)


    body_indented = _normalize_indent(body, target_indent=4)

    return imports + RUN_TASK_INTERFACE + body_indented



def postprocess_continuation_code(llm_generated: str, subtask_name: Optional[str] = None) -> str:



    raw = (llm_generated or "").strip()
    if not raw:

        return ""


    raw = raw.replace("{{", "{").replace("}}", "}")


    cleaned = _clean_llm_text_to_python_code(raw)
    cleaned = _strip_known_marker_blocks(cleaned)




    markers = [
        "Now write the fixed continuation:",
        "Continue from here (add the remaining steps):",
        "Continue from here:",
        "Now write the fixed continuation",
        "# Now write the fixed continuation:",
        "# Continue from here",
    ]
    cut_pos = -1
    for m in markers:
        p = cleaned.rfind(m)
        if p != -1:
            cut_pos = max(cut_pos, p + len(m))
    if cut_pos != -1:
        cleaned = cleaned[cut_pos:].strip()




    body = cleaned
    try:
        t0 = ast.parse(cleaned)

        if any(isinstance(n, ast.FunctionDef) and n.name == "run_task" for n in t0.body):
            body = _extract_function_body_ast(cleaned, "run_task").rstrip("\n")

        elif subtask_name and any(isinstance(n, ast.FunctionDef) and n.name == subtask_name for n in t0.body):
            body = _extract_function_body_ast(cleaned, subtask_name).rstrip("\n")

        elif any(isinstance(n, ast.FunctionDef) and n.name == "run_skill" for n in t0.body):
            body = _extract_function_body_ast(cleaned, "run_skill").rstrip("\n")

    except Exception:
        body = cleaned




    bad_prefixes = ("def ", "def\t", "import ", "from ", "class ", "if __name__")
    drop_exact_prefixes = (
        "# [NEW STATEMENTS", "# [EXECUTED", "# --- executed", "# --- end executed",
        "# [ALREADY EXECUTED", "# [END OF EXECUTED",
        "# SCENE DESCRIPTION", "# CURRENT SCENE", "# FAILED_STATEMENT", "# CONSTRAINTS",
        "# [ERROR]", "# REFERENCE CODE", "# RULES", "# PRIMITIVES", "# UTILITIES",
        "# EXAMPLE", "# HARD CONSTRAINTS", "# LAST_STEP", "# Continue from here",
    )
    ALLOW_CALL_PREFIX = (
        "move_gripper_to(", "move_to_position(", "move_parallel(", "rotate_gripper(",
        "open_gripper(", "close_gripper(", "pick(", "place(", "grasp_handle(", "release_handle(",
        "open_panda(", "close_panda(",
        "open_robotiq85(", "close_robotiq85(", "pick_robotiq85(", "place_robotiq85(",
        "grasp_handle_robotiq85(", "release_handle_robotiq85(",
        "activate_vacuum(", "deactivate_vacuum(", "attach_vacuum_handle(", "detach_vacuum_handle(",
        "activate_suction(", "deactivate_suction(", "attach_suction(", "detach_suction(",
    )

    lines = body.splitlines()


    i = 0

    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            continue
        if s.startswith("#"):

            maybe = _decomment_and_strip_list_prefix(s)


            if maybe.startswith(ALLOW_CALL_PREFIX):

                break

            i += 1
            continue
        break

    lines = lines[i:]


    kept = []
    skip_json_block = False
    json_brace_depth = 0

    for ln in lines:
        s = ln.lstrip()
        st = s.strip()

        if not st:
            kept.append(ln)
            continue


        if st.startswith("```") or st.startswith("[") or st.startswith("]"):
            continue


        if st.startswith("#"):

            if any(st.startswith(pfx) for pfx in drop_exact_prefixes):

                continue
            if st.startswith("# [") and "]" in st:

                continue


            code_like = _decomment_and_strip_list_prefix(st)

            if code_like.startswith(ALLOW_CALL_PREFIX):
                indent = ln[:len(ln) - len(s)]
                kept.append(indent + code_like)

                continue



            continue



        json_keys = [
            "executed_prefix_count", "must_keep_executed_prefix",
            "step_index", "skill_call_count", "skill_name", "need_more_steps",
            '"success":', '"statement":'
        ]
        if any(key in st for key in json_keys):

            skip_json_block = True

            json_brace_depth += st.count("{") - st.count("}")
            continue

        if skip_json_block:
            json_brace_depth += st.count("{") - st.count("}")
            if json_brace_depth <= 0:
                skip_json_block = False
                json_brace_depth = 0
            continue


        if any(st.startswith(bp) for bp in bad_prefixes):
            continue


        if st.startswith("return"):
            indent = ln[:len(ln) - len(s)]
            kept.append(indent + "pass")
            continue


        if st == "..." or st == "…":
            continue

        kept.append(ln)


    body = "\n".join(kept).strip("\n")

    if not body.strip():

        return ""
    body = _strip_prompt_blocks_and_docstrings(body)

    body = _strip_leading_triple_quote_block(body)

    body = rewrite_numpy_max_patterns_in_body(body)



    body = _dedent_continuation(body)
    if not body.strip():

        return ""


    def _is_valid_body(snippet: str) -> bool:
        try:
            wrapped = "def __tmp__():\n" + _normalize_body_indent(snippet, indent=4)
            ast.parse(wrapped)
            return True
        except Exception as e:

            return False

    if _is_valid_body(body):

        return body




    lines2 = body.splitlines()
    for end in range(len(lines2), 0, -1):
        cand = "\n".join(lines2[:end]).rstrip() + "\n"
        if cand.strip() and _is_valid_body(cand):

            return cand



    return body



def preprocess_reference_code(code: str) -> str:
    """
    Reference code에서 첫 번째(top-level) 함수의 body만 추출해서 그대로 반환합니다.
    - import/from 제거할 필요 없음(함수 body만 뽑기 때문)
    - 라인 사이에 # 삽입 없음
    - docstring 포함(원하시면 제거 옵션도 아래에 제공)
    """
    if not code or not code.strip():
        return ""

    src = code.strip() + "\n"

    try:
        tree = ast.parse(src)
    except Exception:

        return _fallback_extract_body_only(src)

    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = node
            break
    if fn is None or not fn.body:
        return ""

    lines = src.splitlines(keepends=True)
    start = fn.body[0].lineno - 1
    end = fn.body[-1].end_lineno
    body = "".join(lines[start:end])


    return textwrap.dedent(body).rstrip()


def _fallback_extract_body_only(src: str) -> str:

    lines = src.splitlines()
    in_def = False
    def_indent = 0
    out = []
    for ln in lines:
        stripped = ln.lstrip()
        if not in_def:
            if stripped.startswith("def ") and (len(ln) - len(stripped) == 0):
                in_def = True
                def_indent = len(ln) - len(stripped)
            continue

        cur_indent = len(ln) - len(stripped)
        if stripped == "":
            out.append("")
            continue
        if cur_indent > def_indent:
            out.append(ln[def_indent+4:] if len(ln) >= def_indent+4 else stripped)
        else:
            break
    return "\n".join(out).rstrip()


"""
def preprocess_reference_code(code: str) -> str:

    if not code:
        return ""

    lines = code.strip().split('\n')
    body_lines = []
    in_function = False
    function_indent = 0

    for line in lines:
        stripped = line.strip()

        # Skip import statements
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue

        # Detect function definition (any def statement that's not a nested function)
        if stripped.startswith('def ') and not in_function:
            in_function = True
            function_indent = len(line) - len(line.lstrip())
            continue

        # If we're in the function, collect the body
        if in_function:
            if line.strip() == '':
                body_lines.append('')
            elif len(line) - len(line.lstrip()) > function_indent:
                # Remove the function's base indentation (4 spaces)
                content = line[function_indent + 4:] if len(line) > function_indent + 4 else line.lstrip()
                body_lines.append(content)
            elif stripped and not stripped.startswith('#'):
                # Hit something at same or lower indent level - function ended
                break

    # Remove trailing empty lines
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()

    # Comment out each line with proper indentation
    commented_lines = []
    indent = "    "  # 4 spaces
    for line in body_lines:
        if line.strip():
            commented_lines.append(f"{line}")
        else:
            commented_lines.append(f"{indent}#")

    return '\n'.join(commented_lines)
"""