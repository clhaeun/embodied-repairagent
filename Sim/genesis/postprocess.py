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
    # "#   [7] foo(...)" -> "foo(...)"
    s = st.lstrip("#").strip()
    s = _LIST_PREFIX.sub("", s).strip()
    return s

def _get_slice_int(slice_node) -> int | None:
    # py3.8: ast.Index(value=ast.Constant); py3.9+: ast.Constant directly
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, int):
        return slice_node.value
    if hasattr(ast, "Index") and isinstance(slice_node, ast.Index):
        v = slice_node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, int):
            return v.value
    return None

def _is_values_call(node: ast.AST) -> bool:
    # something.values()
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

        # node.value should be Call(max(...))
        if not isinstance(node.value, ast.Call):
            return node
        call = node.value

        # call.func should be Name("max")
        if not (isinstance(call.func, ast.Name) and call.func.id == "max"):
            return node

        # need 1 arg: X.values()
        if len(call.args) != 1 or not _is_values_call(call.args[0]):
            return node

        # slice must be int k
        k = _get_slice_int(node.slice)
        if k is None:
            return node

        # 이미 key가 있으면 건드리지 않음
        for kw in call.keywords:
            if kw.arg == "key":
                return node

        # key=lambda p: float(p[k])
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
        return code  # 파싱 안되면 원본 유지

    new_tree = _MaxValuesSubscriptKeyFixer().visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        return ast.unparse(new_tree).rstrip() + "\n"
    except Exception:
        # unparse 실패 시 원본 유지

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
        return body  # rewrite 실패 시 원본 유지

    # 다시 __tmp__ 바디만 뽑아오기
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
        # 혹시 constraints/reference를 마커로 감싸면 여기도 추가
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
        "[REFERENCE CODE",   # "[REFERENCE CODE (TASK...)" 포함
    )
    end_markers = (
        "[EXECUTED_STATEMENTS_END]",
        "[END OF EXECUTED CODE]",
        "[FAILED_STATEMENT_END]",
    )

    for ln in lines:
        s = ln.strip()
        s2 = s.lstrip("#").strip()   # "# [REFERENCE CODE" -> "[REFERENCE CODE"

        # (1) marker block drop  (주석(#)이 붙어도 인식)
        if any(s2.startswith(b) for b in begin_markers):
            in_marker = True
            continue
        if any(s2.startswith(e) for e in end_markers):
            in_marker = False
            continue
        if in_marker:
            continue

        # (2) triple-quote block drop ("""...""" or '''...''')
        if not in_triple and (s.startswith('"""') or s.startswith("'''")):
            in_triple = True
            triple_quote = s[:3]
            # same-line close
            if s.count(triple_quote) >= 2:
                in_triple = False
                triple_quote = None
            continue
        elif in_triple:
            if triple_quote and triple_quote in s:
                in_triple = False
                triple_quote = None
            continue

        # (3) prompt-ish header comments drop
        if s.startswith("# [") and "]" in s:
            continue

        out.append(ln)

    return "\n".join(out)



def _leading_spaces(s: str) -> int:
    return len(s) - len(s.lstrip(" "))
    
TOOL_CALL_PATTERNS = [
    r"<tool_call\b.*?>.*?</tool_call>",   # 블록 형태
    r"</?tool_call\b.*?>",                # 태그만
    r"<\|tool_call\|>.*?<\|/tool_call\|>", # 특수토큰 변형(혹시)
    r"<\|.*?\|>",                          # 기타 모델 특수토큰(보수적으로)
]
def _strip_tool_calls(s: str) -> str:
    # 1) 블록/태그 형태 제거
    for p in TOOL_CALL_PATTERNS:
        s = re.sub(p, "", s, flags=re.DOTALL | re.IGNORECASE)

    # 2) 라인 중간에 붙은 "<tool_call" 이후를 잘라내기 (핵심)
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
    # Use rstrip() to preserve leading indentation of first line
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

    # 공백 아닌 라인만 모아서 최소 들여쓰기 계산
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
            out.append(prefix + ln[min_lead:])  # min만 제거 -> 상대 들여쓰기 유지
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

_BLOCK_OPEN_RE = re.compile(r".*:\s*(#.*)?$")  # 줄 끝에 : 로 블록 여는지

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
    stack = []  # (block_indent, expected_child_indent)
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

        # 5) else/elif/except/finally는 이전 블록과 같은 레벨로 정렬
        if is_else_like and stack:
            # Pop current block and align with previous block's indent
            stack.pop()
            if stack:
                # If nested, use parent's child indent
                block_indent = stack[-1][1]
            else:
                # Top level
                block_indent = 0
            ln = " " * block_indent + stripped
            cur = block_indent

        # 1) 스택 정리: 현재 indent가 블록 indent 이하로 떨어지면 블록 종료
        elif not is_else_like:
            while stack and cur <= stack[-1][0]:
                stack.pop()

        # 2) 블록 컨텍스트에 따라 indent 조정
        if stack and not is_else_like:
            # 블록 내부인 경우: expected_child_indent를 초과하면 줄임
            _, expect = stack[-1]
            if cur > expect:
                ln = " " * expect + stripped
                cur = expect
        elif not is_else_like:
            # 블록 밖인데 이전보다 indent가 증가했으면 -> 이전 레벨로 복구
            if cur > prev_indent:
                ln = " " * prev_indent + stripped
                cur = prev_indent

        out.append(ln)

        # 3) return 문이면 모든 블록 컨텍스트 종료
        if is_return:
            stack.clear()
            prev_indent = 0  # 최상위 레벨로 리셋

        # 4) 현재 줄이 블록을 여는 경우 스택에 push (주석은 제외)
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
    # 주석/빈줄은 baseline 계산에서 제외
    candidates = [ln for ln in non_empty if not ln.lstrip().startswith("#")]

    # 전부 주석뿐이면 그냥 일반 dedent
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

    # body indent 추정
    body_indent = 4
    if fn.body:
        first_body_line_idx = fn.body[0].lineno - 1
        if 0 <= first_body_line_idx < len(lines):
            body_indent = _leading_spaces(lines[first_body_line_idx]) or 4

    # 삽입 라인 결정(1-based lineno)
    insert_lineno = None
    if insert_after_lineno is not None:
        insert_lineno = int(insert_after_lineno) + 2  # 0-based -> 다음 줄 1-based
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

    # ---- 안전장치: 함수 바디 영역 밖으로 삽입 방지 ----
    fn_start_0 = fn.lineno - 1
    fn_end_0 = getattr(fn, "end_lineno", fn.lineno) - 1
    # def 라인 바로 다음부터 함수 끝 라인까지를 안전 범위로
    insert_idx = max(fn_start_0 + 1, min(insert_idx, fn_end_0 + 1))

    cont_norm = _normalize_body_indent(cont_body, indent=body_indent).rstrip("\n")
    if not cont_norm.strip():
        return current_code

    # ---- 안전장치: 앞뒤로 개행 1개씩 확보 ----
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
        print(f"[DEBUG][{tag}] current_code unparsable: {e}")
        return

    ex = RunTaskExtractor(subtask_name)
    ex.visit(tree)

    body = ex.subtask_name_body or []
    next_index = len(executed_stmt_texts)
    remaining = body[next_index:]

    print(f"[DEBUG][{tag}] subtask={subtask_name} next_index={next_index}, body_len={len(body)}, remaining={len(remaining)}")

    for i, stmt in enumerate(remaining[:k], start=next_index + 1):
        try:
            stmt_code = get_statement_code(stmt, current_lines).rstrip()
        except Exception as e:
            stmt_code = f"<failed to render stmt: {e}>"

        kind = "RETURN" if isinstance(stmt, ast.Return) else type(stmt).__name__
        head = stmt_code.splitlines()[0] if stmt_code else ""
        lno = getattr(stmt, "lineno", None)
        end = getattr(stmt, "end_lineno", lno)

        print(f"[DEBUG][{tag}] to_exec[{i}] {kind} (lineno={lno}, end={end}): {head}")

        if show_full:
            print(f"[DEBUG][{tag}] to_exec[{i}] full:\n{stmt_code}\n---")

    if len(remaining) > k:
        print(f"[DEBUG][{tag}] ... ({len(remaining) - k} more statements)")


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
    # 아주 싼 사전 필터: 해당 패턴이 없으면 파싱 자체를 안 함
    if "] + [" not in stmt_code and "]+[" not in stmt_code:
        return stmt_code

    src = textwrap.dedent(stmt_code).strip()
    if not src:
        return stmt_code

    try:
        tree = ast.parse(src)
    except Exception:
        return stmt_code  # 파싱 실패면 원본 유지(보수적)

    new_tree = _ListAddToNpArray().visit(tree)
    ast.fix_missing_locations(new_tree)

    # Python 3.9+ : ast.unparse 사용 가능
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

        # 주석이 아니면 그대로 유지
        if not stripped.startswith("#"):
            result.append(ln)
            continue

        # 주석 제거 후 skill인지 확인
        uncommented = stripped[1:].strip()

        # skill 호출 패턴인 경우 주석 제거
        if any(uncommented.startswith(prefix) for prefix in SKILL_PREFIXES):
            # 원래 들여쓰기 유지
            indent = ln[:len(ln) - len(ln.lstrip())]
            result.append(indent + uncommented)
            print(f"[DEBUG][UNCOMMENT_SKILL] Converted: {stripped[:60]} -> {uncommented[:60]}")
        else:
            # skill이 아닌 주석은 버림 (예: # floor: pos=...)
            print(f"[DEBUG][UNCOMMENT_SKILL] Dropped non-skill comment: {stripped[:60]}")
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
        # 1) run_task가 있으면 run_task body 우선
        if any(isinstance(n, ast.FunctionDef) and n.name == "run_task" for n in t0.body):
            body = _extract_function_body_ast(cleaned, "run_task").rstrip("\n")

        # 2) 그 외에 subtask_name 함수가 있으면 그 body
        elif any(isinstance(n, ast.FunctionDef) and n.name == subtask_name for n in t0.body):
            body = _extract_function_body_ast(cleaned, subtask_name).rstrip("\n")

        # 3) 없으면 cleaned 전체(body-only로 간주)
    except Exception:
        body = cleaned

    # Skill 주석을 코드로 변환
    body = _uncomment_skill_calls(body)

    # Simple and safe indentation normalization
    print(f"[DEBUG][POSTPROCESS_INIT] Before normalize_indent: {len(body)} chars")
    print(f"[DEBUG][POSTPROCESS_INIT] First 300 chars:\n{body[:300]}")

    if not body.strip():
        body = "return True"

    # -------------------------
    # header + interface
    # -------------------------
    robot_type = normalize_robot_type(robot_type)
    imports = HEADER[robot_type].strip("\n") + "\n\n"

    # Use hardcoded signature from SUBTASK_SIGNATURES dictionary
    if subtask_name in SUBTASK_SIGNATURES:
        signature_line, description = SUBTASK_SIGNATURES[subtask_name]
        RUN_TASK_INTERFACE = signature_line + "\n"
        print(f"[DEBUG][POSTPROCESS_INIT] Using hardcoded signature for {subtask_name}: {signature_line}")
    elif run_task_interface and run_task_interface.lstrip().startswith("def ") and run_task_interface.rstrip().endswith(":"):
        # Fallback to reference signature if provided (for backward compatibility)
        RUN_TASK_INTERFACE = run_task_interface.rstrip() + "\n"
        print(f"[DEBUG][POSTPROCESS_INIT] Using reference signature for {subtask_name}: {run_task_interface.rstrip()}")
    else:
        # Last resort: generic signature
        RUN_TASK_INTERFACE = "def {}(env, obj_name=None, target_name=None, **kwargs):\n".format(subtask_name)
        print(f"[WARN][POSTPROCESS_INIT] Using generic signature for {subtask_name}")

    # Conditionally apply numpy pattern rewrite only if ast.unparse is available (Python 3.9+)
    if hasattr(ast, "unparse"):
        body = rewrite_numpy_max_patterns_in_code(body)

    # Normalize indentation (safe: only adjusts minimum indent to target_indent=4)
    body_indented = _normalize_indent(body, target_indent=4)
    print(f"[DEBUG][POSTPROCESS_INIT] After normalize_indent: {len(body_indented)} chars")
    return imports + RUN_TASK_INTERFACE + body_indented



def postprocess_continuation_code(llm_generated: str, subtask_name: Optional[str] = None) -> str:
    print(f"[DEBUG][POSTPROCESS_CONT] Input length: {len(llm_generated or '')}")
    print(f"[DEBUG][POSTPROCESS_CONT] subtask_name: {subtask_name}")

    raw = (llm_generated or "").strip()
    if not raw:
        print("[DEBUG][POSTPROCESS_CONT] Empty input - returning empty string")
        return ""

    # 0) 흔한 문법 파괴 토큰 방어
    raw = raw.replace("{{", "{").replace("}}", "}")

    # 1) 공통 클리닝
    cleaned = _clean_llm_text_to_python_code(raw)
    cleaned = _strip_known_marker_blocks(cleaned)

    # ------------------------------------------------------------
    # A) 프롬프트 에코 방지: "마커 이후"만 취하기 (가장 마지막 마커 기준)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # B) AST로 함수 바디 추출 (가능하면)
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # C) "코드 라인만" 남기기: 주석/헤더/markdown/constraints/json 덩어리 제거
    # ------------------------------------------------------------
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

    # (1) 맨 앞에서부터 "첫 번째 비주석/비공백 코드 라인"이 나올 때까지 주석/빈줄 제거
    i = 0
    print(f"[DEBUG][POSTPROCESS_CONT] Starting first loop, total lines: {len(lines)}")
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            i += 1
            continue
        if s.startswith("#"):
            # 주석 해제 + list prefix 제거해서 코드인지 확인
            maybe = _decomment_and_strip_list_prefix(s)
            print(f"[DEBUG][POSTPROCESS_CONT] Line {i}: '{s[:80]}' -> decommented: '{maybe[:80]}'")
            #  코드처럼 생긴 주석이면 여기서 멈춰서 아래 루프가 처리하게 함
            if maybe.startswith(ALLOW_CALL_PREFIX):
                print(f"[DEBUG][POSTPROCESS_CONT] Found code-like comment at line {i}: {s[:80]}")
                break
            print(f"[DEBUG][POSTPROCESS_CONT] Not code-like, skipping line {i}")
            i += 1
            continue
        break
    print(f"[DEBUG][POSTPROCESS_CONT] Skipped {i} lines at beginning, remaining: {len(lines) - i}")
    lines = lines[i:]

    # (2) 본문에서 프롬프트 잔재 제거 + constraints json 블록 제거
    kept = []
    skip_json_block = False
    json_brace_depth = 0

    for ln in lines:
        s = ln.lstrip()
        st = s.strip()

        if not st:
            kept.append(ln)
            continue

        # 마크다운/펜스/잡텍스트 제거
        if st.startswith("```") or st.startswith("[") or st.startswith("]"):
            continue

        # prompt 헤더류 주석 제거
        if st.startswith("#"):
            # 1) 프롬프트 헤더류는 기존대로 drop
            if any(st.startswith(pfx) for pfx in drop_exact_prefixes):
                print(f"[DEBUG][POSTPROCESS_CONT] Dropping prompt header: {st[:60]}")
                continue
            if st.startswith("# [") and "]" in st:
                print(f"[DEBUG][POSTPROCESS_CONT] Dropping bracket comment: {st[:60]}")
                continue

            # 2) 코드처럼 생긴 주석이면 주석 해제해서 살림
            code_like = _decomment_and_strip_list_prefix(st)
            print(f"[DEBUG][POSTPROCESS_CONT] Checking comment: '{st[:80]}' -> code_like: '{code_like[:80]}'")
            if code_like.startswith(ALLOW_CALL_PREFIX):
                indent = ln[:len(ln) - len(s)]
                kept.append(indent + code_like)
                print(f"[DEBUG][POSTPROCESS_CONT] KEPT code-like comment: {code_like[:60]}")
                continue

            # 3) 나머지 주석은 drop(원하시면 kept.append(ln)로 "보존"도 가능)
            print(f"[DEBUG][POSTPROCESS_CONT] Dropping non-code comment: {st[:60]}")
            continue

        # constraints json 같은 덩어리 제거 (키가 보이면 그 블록 통째로 버림)
        # LAST_STEP JSON 블록도 제거 (step_index, skill_call_count, skill_name 등)
        json_keys = [
            "executed_prefix_count", "must_keep_executed_prefix",
            "step_index", "skill_call_count", "skill_name", "need_more_steps",
            '"success":', '"statement":'
        ]
        if any(key in st for key in json_keys):
            # 그 라인 포함해서 블록 제거 시작
            skip_json_block = True
            # 중괄호 깊이 추적
            json_brace_depth += st.count("{") - st.count("}")
            continue

        if skip_json_block:
            json_brace_depth += st.count("{") - st.count("}")
            if json_brace_depth <= 0:
                skip_json_block = False
                json_brace_depth = 0
            continue

        # continuation 계약 위반 제거
        if any(st.startswith(bp) for bp in bad_prefixes):
            continue

        # return은 continuation에서 위험 -> pass로 치환
        if st.startswith("return"):
            indent = ln[:len(ln) - len(s)]
            kept.append(indent + "pass")
            continue

        # '...' 같은 불완전 토큰 제거
        if st == "..." or st == "…":
            continue

        kept.append(ln)


    body = "\n".join(kept).strip("\n")
    print(f"[DEBUG][POSTPROCESS_CONT] After filtering lines: {len(body)} chars, {len(kept)} lines kept")
    if not body.strip():
        print("[DEBUG][POSTPROCESS_CONT] Body empty after filtering lines - returning empty string")
        return ""
    body = _strip_prompt_blocks_and_docstrings(body)
    print(f"[DEBUG][POSTPROCESS_CONT] After strip_prompt_blocks: {len(body)} chars")
    body = _strip_leading_triple_quote_block(body)
    print(f"[DEBUG][POSTPROCESS_CONT] After strip_triple_quote: {len(body)} chars")
    body = rewrite_numpy_max_patterns_in_body(body)
    print(f"[DEBUG][POSTPROCESS_CONT] After rewrite_numpy_max: {len(body)} chars")

    # 4) dedent
    body = _dedent_continuation(body)
    if not body.strip():
        print("[DEBUG][POSTPROCESS_CONT] Body empty after dedent - returning empty string")
        return ""

    # 5) 문법 검증
    def _is_valid_body(snippet: str) -> bool:
        try:
            wrapped = "def __tmp__():\n" + _normalize_body_indent(snippet, indent=4)
            ast.parse(wrapped)
            return True
        except Exception as e:
            print(f"[DEBUG][POSTPROCESS_CONT] Validation failed: {type(e).__name__}: {e}")
            return False

    if _is_valid_body(body):
        print(f"[DEBUG][POSTPROCESS_CONT] Valid body - returning {len(body)} chars")
        return body

    # 6) (기존 복구 로직은 유지하되, 여기까지 왔으면 보통 LLM이 깨진 블록을 준 케이스)
    #    끝에서부터 truncate 하며 최대 valid prefix 찾기
    print("[DEBUG][POSTPROCESS_CONT] Body invalid - trying to recover by truncating")
    lines2 = body.splitlines()
    for end in range(len(lines2), 0, -1):
        cand = "\n".join(lines2[:end]).rstrip() + "\n"
        if cand.strip() and _is_valid_body(cand):
            print(f"[DEBUG][POSTPROCESS_CONT] Recovered valid prefix with {end} lines")
            return cand

    print("[DEBUG][POSTPROCESS_CONT] Could not recover valid body - returning as-is")
    print("[DEBUG][POST] final continuation lines =", len(body.splitlines()))
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
        # AST가 깨진 경우: 기존처럼 라인 기반 fallback (간단)
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

    # body를 "그대로" 쓰되, 최소 들여쓰기만 제거해서 깔끔하게
    return textwrap.dedent(body).rstrip()


def _fallback_extract_body_only(src: str) -> str:
    # ast.parse 실패 시: 가장 단순한 라인 기반 추출
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
        # 함수 바디: def_indent보다 깊은 줄만 수집
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