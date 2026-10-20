"""
RLBench Loop - AST 기반 step-by-step 또는 failure-only 실행 루프

두 가지 모드:
1. step 모드: run_skill 함수의 각 statement 실행 후마다 scene 정보를 JSON으로 출력
2. failure 모드: run_skill을 쭉 실행하다가 failure 발생 시에만 JSON 출력

Usage:
    python rlbench_loop.py --task BasketballInHoop --mode step --source_robot panda --target_robot ur5
    python rlbench_loop.py --task BasketballInHoop --mode failure --source_robot panda --target_robot ur5
    python rlbench_loop.py --task BasketballInHoop --mode step --max_steps 5  # 최대 5번의 skill 호출로 제한
"""
import re
import argparse
import importlib
import ast
import astor
import sys
import json
import traceback
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from env import setup_environment, shutdown_environment
from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from rlbench.backend.exceptions import InvalidActionError
from utils.helper import object_names as get_object_names, to_camel_case, list_task_objects


from policy_provider import PolicyProvider
from prompts.prompt_utils import get_prompt

from rlbench.gym import RLBenchEnv

from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor

from grasp_info import get_graspable_objects_info, is_graspable_object, check_gripper_alignment_for_scene
from exploration_utils import compute_static_observation


def patched_close(self) -> None:
    pass

RLBenchEnv.close = patched_close






def numpy_to_list(obj):
    """numpy 배열을 리스트로 변환 (JSON 직렬화용)"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj

from headers import RUN_SKILL_INTERFACE, HEADER



from typing import Iterable

def _remove_kwarg(code: str, fn_names: Iterable[str], kw: str) -> str:
    """
    주어진 함수 호출들(fn_names) 안에서 특정 키워드 인자(kw=...)를 제거
    포맷(줄바꿈/들여쓰기)은 최대한 유지
    """
    for fn in fn_names:

        code = re.sub(
            rf"({re.escape(fn)}\s*\(.*?),(?:\s|\n|\r|\t)*{re.escape(kw)}\s*=\s*[^,\)]*(?=\s*\))",
            r"\1",
            code,
            flags=re.DOTALL
        )

        code = re.sub(
            rf"({re.escape(fn)}\s*\(.*?),(?:\s|\n|\r|\t)*{re.escape(kw)}\s*=\s*[^,\)]*(?=\s*,)",
            r"\1",
            code,
            flags=re.DOTALL
        )

        code = re.sub(
            rf"({re.escape(fn)}\s*\()(?:\s|\n|\r|\t)*{re.escape(kw)}\s*=\s*[^,\)]*\s*,",
            r"\1",
            code,
            flags=re.DOTALL
        )
    return code


_TIMEOUT_TARGET_FNS = [
    "sawyer_move_to", "sawyer_open_gripper", "sawyer_close_gripper",
    "sawyer_pick", "sawyer_place",
]

def remove_invalid_kwargs(code: str) -> str:
    """
    sawyer 함수들에서 지원하지 않는 키워드 인자를 일괄 제거
    - timeout_s / timeout: sawyer_move_to, sawyer_open_gripper, sawyer_close_gripper, sawyer_pick, sawyer_place
    - approach_axis / approach_distance: sawyer_place
    """

    code = _remove_kwarg(code, _TIMEOUT_TARGET_FNS, "timeout_s")
    code = _remove_kwarg(code, _TIMEOUT_TARGET_FNS, "timeout")

    code = _remove_kwarg(code, ["sawyer_place"], "approach_axis")
    code = _remove_kwarg(code, ["sawyer_place"], "approach_distance")
    return code

def _remove_is_triggered_calls(code: str) -> str:
    """
    Replace .is_triggered() calls with 'done', as ProximitySensor does not have this method.
    e.g. 'if success.is_triggered():' -> 'if done:'
    This prevents AttributeError: 'ProximitySensor' object has no attribute 'is_triggered'
    """
    import re
    code = re.sub(r'[A-Za-z_][A-Za-z0-9_]*\.is_triggered\(\)', 'done', code)
    return code


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

def _pick_best_code_block(s: str) -> str | None:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if not blocks:
        return None

    valid_blocks = []
    for b in blocks:
        stripped = b.strip()

        if len(stripped) < 50:
            continue
        if not any(indicator in stripped for indicator in ['=', '(', 'def ', 'import ', 'from ']):
            continue
        valid_blocks.append(stripped)

    if not valid_blocks:
        return None


    for b in valid_blocks:
        if re.search(r"^\s*def\s+run_skill\s*\(", b, flags=re.MULTILINE):
            return b


    return max(valid_blocks, key=len)

def _clean_llm_text_to_python_code(raw: str) -> str:
    s = (raw or "").lstrip("\n\r").rstrip()

    sm, em = "[INITIAL_CODE_START]", "[INITIAL_CODE_END]"
    if sm in s and em in s:
        s = s[s.index(sm) + len(sm): s.index(em)].lstrip("\n\r").rstrip()
    else:
        picked = _pick_best_code_block(s)
        if picked is not None:
            s = picked


    s = re.sub(r"```(?:python)?", "", s, flags=re.IGNORECASE).replace("```", "").lstrip("\n\r").rstrip()


    s = _strip_tool_calls(s)
    s = s.replace("\t", "    ")


    s = re.sub(r"^\s*obj\s*\[.*\]\s*$", "", s, flags=re.MULTILINE)

    return s.lstrip("\n\r").rstrip() + "\n"


def _fix_variable_name_typos(code: str) -> str:
    """LLM이 자주 틀리는 변수명을 수정합니다.
    - variation_index → variations_index (함수 시그니처와 일치)
    """

    code = code.replace("variation_index", "variations_index")
    return code


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


class _FixAlignGripperKwargs(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)


        func = node.func
        is_align = {isinstance(func, ast.Name) 
        and func.id in ("ur5_align_gripper", "sawyer_align_gripper", "jaco_align_gripper")}
        if not is_align:
            return node


        for kw in node.keywords:
            if kw.arg == "reference_quaternion":
                kw.arg = "reference_quat"
        return node

def fix_quaternion(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    tree = _FixAlignGripperKwargs().visit(tree)
    ast.fix_missing_locations(tree)

    try:
        return ast.unparse(tree) + "\n"
    except Exception:
        return code

def _normalize_indent(body: str, target_indent: int = 4) -> str:
    lines = (body or "").splitlines()


    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return "\n"

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))


    code_lines = [ln for ln in non_empty if not ln.lstrip().startswith("#")]
    ref_lines = code_lines if code_lines else non_empty


    from collections import Counter
    lead_counts = Counter(lead(ln) for ln in ref_lines)
    base_lead = lead_counts.most_common(1)[0][0]

    prefix = " " * target_indent
    out = []
    for ln in lines:
        if ln.strip():
            ln_lead = lead(ln)

            relative = max(0, ln_lead - base_lead)
            relative = round(relative / target_indent) * target_indent
            out.append(prefix + " " * relative + ln.lstrip(" "))
        else:
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def postprocess_initial_code(llm_generated: str, robot_name: str) -> str:
    """
    - raw에서 코드만 정제
    - top-level에 run_skill이 "진짜로" 있으면(AST) body만 추출
    - 없으면 body-only로 간주
    - (추가 안전) body 안에 중첩 def run_skill이 있으면 안쪽 body로 교체
    - 최종적으로 header + RUN_SKILL_INTERFACE + body 구성
    """
    raw = (llm_generated or "").strip()
    cleaned = _clean_llm_text_to_python_code(raw)
    cleaned = cleaned.replace("reference_quaternion", "reference_quat")
    cleaned = remove_invalid_kwargs(cleaned)
    cleaned = _fix_variable_name_typos(cleaned)


    cleaned = _remove_is_triggered_calls(cleaned)



    has_top_run_skill = False
    try:
        t0 = ast.parse(cleaned)
        has_top_run_skill = any(
            isinstance(n, ast.FunctionDef) and n.name == "run_skill"
            for n in t0.body
        )
    except Exception:
        has_top_run_skill = False



    if has_top_run_skill:
        try:
            body = _extract_function_body_ast(cleaned, "run_skill")
            if not body or not body.strip():
                body = cleaned
        except Exception:
            body = cleaned
    else:
        body = cleaned


    if "def run_skill" in body:
        try:
            wrapped = "def __tmp__():\n" + _normalize_indent(body, target_indent=4)
            t1 = ast.parse(wrapped)
            tmp_fn = t1.body[0]

            inner = None
            for st in tmp_fn.body:
                if isinstance(st, ast.FunctionDef) and st.name == "run_skill":
                    inner = st
                    break

            if inner is not None and inner.body:
                new_lines = []
                for st in inner.body:
                    try:
                        new_lines.append(ast.unparse(st))
                    except Exception:
                        pass
                body = "\n".join([ln for ln in new_lines if ln.strip()]).rstrip() + "\n"
        except Exception:
            pass


    _real_lines = [
        ln for ln in body.splitlines()
        if ln.strip()
        and not ln.lstrip().startswith(("def run_skill", "import ", "from ", '"""', "'''"))
    ]
    if not _real_lines:
        raise ValueError("LLM returned empty run_skill body (only def/import lines, no actual code)")

    body = _normalize_indent(body)
    body = fix_quaternion(body)

    header = HEADER.get(robot_name, HEADER["panda"])
    full_code = header + "\n\n" + RUN_SKILL_INTERFACE + body




    return full_code



def get_scene_info(task, task_name: str = "", robot_type: str = "panda", include_grasp_guidance: bool = True) -> Dict[str, Any]:
    """현재 scene의 object 정보를 반환

    Args:
        task: RLBench task environment
        task_name: Name of the current task
        robot_type: Type of robot (panda, ur5, sawyer, jaco)
        include_grasp_guidance: Whether to include grasp guidance for graspable objects

    Returns:
        Dictionary containing objects info, gripper info, and optionally grasp guidance
    """
    _type_registry = {'Shape': Shape, 'Joint': Joint, 'Dummy': Dummy, 'ProximitySensor': ProximitySensor}

    objects_info = []
    for o in list_task_objects(task):
        name = o.get_name()
        type_str = str(o.get_type())
        type_name = to_camel_case(type_str.split(".")[-1])


        if any(sub in name for sub in ('waypoint', 'visual', 'spawn', 'wrap', 'topPlate', 'boundary', 'base', 'dynamic')):
            if name != 'dirt_boundary':
                continue

        try:
            cls = _type_registry.get(type_name)
            if cls:
                obj = cls(name)
                pos = obj.get_position()
                quat = obj.get_quaternion()
                objects_info.append({
                    "name": name,
                    "type": type_name,
                    "position": numpy_to_list(pos),
                    "quaternion": numpy_to_list(quat)
                })
        except Exception:
            objects_info.append({
                "name": name,
                "type": type_name,
                "position": None,
                "quaternion": None
            })


    obs = task.get_observation()
    gripper_info = {
        "position": numpy_to_list(obs.gripper_pose[:3]),
        "quaternion": numpy_to_list(obs.gripper_pose[3:7]),
        "is_open": bool(obs.gripper_open > 0.5)
    }

    result = {
        "objects": objects_info,
        "gripper": gripper_info
    }


    if include_grasp_guidance:
        try:
            grasp_infos, grasp_guidance_text = get_graspable_objects_info(
                objects_info, task_name, robot_type
            )
            result["grasp_guidance"] = grasp_guidance_text
            result["graspable_objects"] = grasp_infos
        except Exception as e:

            result["grasp_guidance"] = ""
            result["graspable_objects"] = []

    return result

def get_initial_info(task, task_name: str, robot_type: str, descriptions: List[str]) -> Dict[str, Any]:
    """초기 환경 정보를 JSON 형식으로 반환"""
    scene_info = get_scene_info(task, task_name, robot_type)


    descriptions_list = [s[0].upper() + s[1:] if s else '' for s in descriptions]
    task_description = ". ".join(descriptions_list) + "." if descriptions_list else ""




    return {
        "type": "initial",
        "task_name": task_name,
        "task_description": task_description,
        "robot_type": robot_type,
        "scene": scene_info
    }

def output_json(data: Dict[str, Any]):
    """JSON 출력 (stdout)"""
    serializable_data = numpy_to_list(data)
    print(f"[JSON_START]\n{json.dumps(serializable_data, indent=2, ensure_ascii=False)}\n[JSON_END]")

def _leading_spaces(s: str) -> int:
    return len(s) - len(s.lstrip(" "))

def _normalize_body_indent(body: str, indent: int) -> str:
    """
    body 조각의 '최소 들여쓰기'를 제거한 뒤, indent만큼 다시 들여쓰기.
    (상대 들여쓰기(블록 내부)는 유지)
    주석과 코드가 섞여있어도 일관되게 처리.
    """
    lines = (body or "").splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return ""


    min_lead = min(_leading_spaces(ln) for ln in non_empty)
    prefix = " " * indent

    out = []
    for ln in lines:
        if ln.strip():


            current_lead = _leading_spaces(ln)
            if current_lead >= min_lead:
                out.append(prefix + ln[min_lead:])
            else:


                out.append(prefix + ln.lstrip())
        else:
            out.append("")
    return "\n".join(out).rstrip() + "\n"

def _dedent_continuation(body: str) -> str:
    """
    Dedent continuation code by removing common leading whitespace.
    Normalizes indentation for both code and comment lines to avoid inconsistencies.

    Uses the MODE (most common indent level) instead of MIN to handle the case
    where the LLM outputs the first line with fewer spaces than the rest.
    e.g.:
      '  chicken = Shape(...)'      # 2 spaces (outlier)
      '     grill = Shape(...)'     # 5 spaces (majority)
      '     pos = grill.get_...'    # 5 spaces (majority)
    With min=2, dedent produces broken indent (0, 3, 3).
    With mode=5, dedent produces correct flat code (0, 0, 0).
    """
    from collections import Counter

    lines = (body or "").splitlines()
    if not any(ln.strip() for ln in lines):
        return ""

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(" "))

    non_empty = [ln for ln in lines if ln.strip()]


    code_lines = [ln for ln in non_empty if not ln.lstrip().startswith("#")]


    ref_lines = code_lines if code_lines else non_empty


    lead_counts = Counter(lead(ln) for ln in ref_lines)
    max_count = lead_counts.most_common(1)[0][1]

    base = min(indent for indent, count in lead_counts.items() if count == max_count)


    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        n = lead(ln)
        if n >= base:
            out.append(ln[base:])
        else:


            out.append(ln.lstrip(" "))
    return "\n".join(out).rstrip() + "\n"

def _find_run_skill_node(tree: ast.AST) -> Optional[ast.FunctionDef]:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.FunctionDef) and node.name == "run_skill":
            return node
    return None

def merge_continuation_into_run_skill(
    current_code: str,
    cont_body: str,
    insert_before_last_return: bool = True,
    insert_after_lineno: Optional[int] = None,
) -> str:
    """
    current_code(전체 코드, def run_skill 포함) 안의 run_skill body에
    cont_body(continuation 조각)를 삽입/추가해서 새 전체 코드를 반환합니다.
    """
    if not (cont_body or "").strip():
        return current_code


    try:
        tree = ast.parse(current_code)
    except Exception:

        return current_code

    fn = _find_run_skill_node(tree)
    if fn is None:

        return current_code

    lines = current_code.splitlines(keepends=True)


    body_indent = 4
    if fn.body:
        first_body_line_idx = fn.body[0].lineno - 1
        if 0 <= first_body_line_idx < len(lines):
            body_indent = _leading_spaces(lines[first_body_line_idx])
            if body_indent <= 0:
                body_indent = 4


    insert_lineno = None
    if insert_after_lineno is not None:
        insert_lineno = int(insert_after_lineno) + 1
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


    cont_norm = _normalize_body_indent(cont_body, indent=body_indent)
    if not cont_norm.strip():
        return current_code


    if insert_idx > 0 and not lines[insert_idx - 1].endswith("\n"):
        lines[insert_idx - 1] += "\n"
    if cont_norm and not cont_norm.endswith("\n"):
        cont_norm += "\n"

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


def remove_truncated_lines(code: str) -> str:
    """
    SyntaxError를 유발하는 잘린(truncated) 라인을 제거합니다.
    반복적으로 ast.parse를 시도하여 SyntaxError가 발생하는 라인을 제거합니다.
    'expected an indented block' 에러인 경우 라인 삭제 대신 pass를 삽입합니다.
    최대 5회 반복 (중첩된 잘린 라인 처리).
    """
    for _ in range(5):
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            if e.lineno is None:
                return code
            lines = code.splitlines()
            err_idx = e.lineno - 1
            if err_idx < 0 or err_idx >= len(lines):
                return code


            err_msg = str(e.msg) if e.msg else ""
            if "expected an indented block" in err_msg:

                prev_idx = err_idx - 1
                if prev_idx >= 0:
                    prev_line = lines[prev_idx]
                    indent = len(prev_line) - len(prev_line.lstrip()) + 4
                else:
                    indent = 4
                pass_line = " " * indent + "pass"
                print(f"[TRUNCATED_LINE_REMOVAL] Inserting 'pass' at line {e.lineno} (indented block missing)")
                lines.insert(err_idx, pass_line)
                code = "\n".join(lines)
                continue

            removed_line = lines[err_idx]
            print(f"[TRUNCATED_LINE_REMOVAL] Removing line {e.lineno}: {removed_line.strip()}")
            lines.pop(err_idx)
            code = "\n".join(lines)
    return code


def postprocess_continuation_code(llm_generated: str) -> str:
    raw = (llm_generated or "").strip()
    if not raw:
        return ""



    raw = raw.replace("{{", "{").replace("}}", "}")


    cleaned = _clean_llm_text_to_python_code(raw)
    cleaned = remove_invalid_kwargs(cleaned)
    cleaned = _fix_variable_name_typos(cleaned)


    cleaned = _remove_is_triggered_calls(cleaned)


    body = cleaned
    if "def run_skill" in cleaned:
        try:
            body = _extract_function_body_ast(cleaned, "run_skill")
        except Exception:
            body = cleaned

    body = (body or "").strip("\n")
    if not body.strip():
        return ""


    bad_prefixes = ("def ", "def\t", "import ", "from ", "class ", "if __name__")

    descriptive_comment_patterns = (
        "# Corrected", "# Fixed", "# Continuation", "# Additional",
        "# Updated", "# Revised", "# Modified", "# New"
    )

    _LLM_ARTIFACT_RE = re.compile(r"^\s*obj\s*\[.*\]\s*$")

    kept_lines = []
    first_code_line_seen = False

    for ln in body.splitlines():
        s = ln.lstrip()


        if any(s.startswith(bp) for bp in bad_prefixes):
            continue
        if s.startswith("```") or s.startswith("[INITIAL_CODE_START]") or s.startswith("[INITIAL_CODE_END]"):
            continue
        if s.startswith("return"):
            indent = ln[:len(ln) - len(s)]
            kept_lines.append(indent + "pass")
            first_code_line_seen = True
            continue


        if _LLM_ARTIFACT_RE.match(ln):
            continue


        if not first_code_line_seen and s.startswith("#"):

            if any(s.startswith(pattern) for pattern in descriptive_comment_patterns):
                continue


        if s and not s.startswith("#"):
            first_code_line_seen = True

        kept_lines.append(ln)

    body = "\n".join(kept_lines).strip("\n")
    if not body.strip():
        return ""


    body = _dedent_continuation(body)
    if not body.strip():
        return ""



    def _is_valid_body(snippet: str) -> bool:
        try:
            wrapped = "def __tmp__():\n" + _normalize_body_indent(snippet, indent=4)
            ast.parse(wrapped)
            return True
        except Exception:
            return False

    if _is_valid_body(body):
        return body


    lines = body.splitlines()
    start = 0
    code_start_re = re.compile(r"^\s*(#|[A-Za-z_][A-Za-z0-9_]*\s*=?|\(|\)|\[|\]|obs\s*,\s*reward\s*,\s*done\s*=)")
    for i, ln in enumerate(lines):
        if code_start_re.search(ln):
            start = i
            break
    body2 = "\n".join(lines[start:]).strip("\n")
    body2 = _normalize_body_indent(body2, indent=0).rstrip() + "\n"
    if body2.strip() and _is_valid_body(body2):
        return body2



    lines2 = body2.splitlines()
    for end in range(len(lines2), 0, -1):
        cand = "\n".join(lines2[:end]).rstrip() + "\n"

        if cand.strip() and _is_valid_body(cand):
            return cand


    picked = _pick_best_code_block(raw)
    if picked:
        picked_clean = _clean_llm_text_to_python_code(picked)
        picked_body = picked_clean
        if "def run_skill" in picked_clean:
            try:
                picked_body = _extract_function_body_ast(picked_clean, "run_skill")
            except Exception:
                picked_body = picked_clean

        kept = []
        for ln in (picked_body or "").splitlines():
            s = ln.lstrip()
            if any(s.startswith(bp) for bp in bad_prefixes):
                continue
            if s.startswith("```"):
                continue
            if s.startswith("return"):
                continue
            kept.append(ln)

        picked_body = "\n".join(kept).strip("\n")
        picked_body = _normalize_body_indent(picked_body, indent=0).rstrip() + "\n"
        if picked_body.strip() and _is_valid_body(picked_body):
            return picked_body


        pl = picked_body.splitlines()
        for end in range(len(pl), 0, -1):
            cand = "\n".join(pl[:end]).rstrip() + "\n"
            if cand.strip() and _is_valid_body(cand):
                return cand



    return body


import textwrap

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

def _apply_list_add_to_np_array_on_body(body: str) -> str:
    """
    body(함수 본문 코드 조각)에 대해:
    - def __tmp__ 로 감싼 뒤 AST 변환(_ListAddToNpArray)
    - 다시 body만 추출해서 반환
    """
    if not (body or "").strip():
        return body

    try:
        wrapped = "def __tmp__():\n" + _normalize_body_indent(body, indent=4)
        tree = ast.parse(wrapped)

        tree = _ListAddToNpArray().visit(tree)
        ast.fix_missing_locations(tree)

        new_src = ast.unparse(tree)
        new_body = _extract_function_body_ast(new_src, "__tmp__")

        new_body = _dedent_continuation(new_body)
        return new_body
    except Exception:

        return body


def _apply_list_add_to_np_array_on_full_code(code: str) -> str:
    """
    전체 코드(HEADER + def run_skill 포함)에 대해 AST 변환 적용.
    """
    if not (code or "").strip():
        return code

    try:
        tree = ast.parse(code)
        tree = _ListAddToNpArray().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return code


def fix_type_mismatches(code: str, type_map: Dict[str, str]) -> str:
    """
    주어진 코드 문자열에서 변수 이름별로 기대되는 생성자 타입(type_map)을 참고하여,
    실제 호출된 생성자가 다르면 기대 타입으로 교체한 코드를 반환한다.
    예: "tv_frame = Joint('tv_frame')" + {'tv_frame': 'Shape'} -> "tv_frame = Shape('tv_frame')"
    """
    class TypeFixer(ast.NodeTransformer):
        def visit_Assign(self, node):

            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return self.generic_visit(node)
            

            key_name = None
            if isinstance(node.value, ast.Call) and node.value.args:

                first_arg = node.value.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    key_name = first_arg.value
                elif isinstance(first_arg, ast.Str):
                    key_name = first_arg.s
            
            if not key_name:
                return self.generic_visit(node)
            
            try:
                expected_type = type_map.get(key_name)
            except Exception as e:
                print(e)
                return self.generic_visit(node)
            
            if expected_type and isinstance(node.value, ast.Call):
                func = node.value.func
                current_type = None
                if isinstance(func, ast.Name):
                    current_type = func.id
                elif isinstance(func, ast.Attribute):
                    current_type = func.attr
                if current_type and current_type != expected_type:

                    if isinstance(func, ast.Name):
                        node.value.func = ast.copy_location(
                            ast.Name(id=expected_type, ctx=ast.Load()), func
                        )
                    elif isinstance(func, ast.Attribute):
                        node.value.func = ast.copy_location(
                            ast.Attribute(value=func.value, attr=expected_type, ctx=ast.Load()), func
                        )
            return self.generic_visit(node)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    fixer = TypeFixer()
    fixed_tree = fixer.visit(tree)
    ast.fix_missing_locations(fixed_tree)
    try:
        fixed_code = ast.unparse(fixed_tree)
    except AttributeError:
        raise RuntimeError("이 환경에서 ast.unparse(fixed_tree)를 사용할 수 없습니다.")
    return fixed_code







class RunSkillExtractor(ast.NodeVisitor):
    """run_skill 함수의 body를 추출하는 AST visitor"""

    def __init__(self):
        self.run_skill_body: List[ast.stmt] = []
        self.run_skill_args: List[str] = []
        self.helper_functions: Dict[str, ast.FunctionDef] = {}
        self.imports: List[ast.stmt] = []

    def visit_Import(self, node):
        self.imports.append(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.imports.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "run_skill":
            self.run_skill_body = node.body
            self.run_skill_args = [arg.arg for arg in node.args.args]
        else:
            self.helper_functions[node.name] = node
        self.generic_visit(node)


def get_statement_code(stmt: ast.stmt, source_lines: List[str]) -> str:
    """AST statement에 해당하는 원본 코드를 반환"""
    start_line = stmt.lineno - 1
    end_line = getattr(stmt, 'end_lineno', stmt.lineno)
    return '\n'.join(source_lines[start_line:end_line])


def is_skill_call(stmt: ast.stmt) -> Tuple[bool, Optional[str]]:
    """statement가 skill 호출인지 확인하고, skill 이름 반환"""
    skill_names = {'pick', 'place', 'move', 'push', 'open_gripper', 'close_gripper',
                   'align_two_axes', 'align_to_quaternion', 'normalize_quaternion',
                   'angle_diff',

                   'ur5_move_to', 'ur5_grasp_at', 'ur5_release_at', 'ur5_align_gripper',
                   'close_ur5_ee', 'open_ur5_ee',

                   'sawyer_move_to', 'sawyer_align_gripper', 'sawyer_open_gripper',
                   'sawyer_close_gripper', 'sawyer_pick', 'sawyer_place'}

    def find_call(node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in skill_names:
                return node.func.id
            elif isinstance(node.func, ast.Attribute) and node.func.attr in skill_names:
                return node.func.attr
        return None


    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        skill = find_call(stmt.value)
        if skill:
            return True, skill


    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        skill = find_call(stmt.value)
        if skill:
            return True, skill


    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        skill = find_call(stmt.value)
        if skill:
            return True, skill

    return False, None


def contains_return(stmt: ast.stmt) -> bool:
    """statement 내부에 return 문이 포함되어 있는지 재귀적으로 확인"""
    if isinstance(stmt, ast.Return):
        return True
    for child in ast.walk(stmt):
        if isinstance(child, ast.Return):
            return True
    return False


def remove_returns_from_stmt(stmt: ast.stmt) -> ast.stmt:
    """statement 내부의 return 문을 pass로 대체"""
    class ReturnRemover(ast.NodeTransformer):
        def visit_Return(self, node):

            return ast.Pass()

    transformer = ReturnRemover()
    return transformer.visit(stmt)






def create_execution_context(env, task, robot_type: str) -> Dict[str, Any]:
    """실행 컨텍스트 생성 (import 및 global 변수들)"""

    if robot_type.lower() == "ur5":
        skill_module_name = "skill_code_ur5"
    elif robot_type.lower() == "sawyer":
        skill_module_name = "skill_code_sawyer"
    elif robot_type.lower() == "jaco":
        skill_module_name = "skill_code_jaco"
    else:
        skill_module_name = "skill_code"

    try:
        skill_module = importlib.import_module(skill_module_name)
    except ImportError:

        skill_module = importlib.import_module("skill_code")


    import env_utils

    context = {
        'env': env,
        'task': task,
        'np': np,
        'Shape': Shape,
        'Joint': Joint,
        'Dummy': Dummy,
        'ProximitySensor': ProximitySensor,

        'get_bbox_sizes': env_utils.get_bbox_sizes,
        'quat_mul': env_utils.quat_mul,
        'normalize_vector': env_utils.normalize_vector,

        'pick': getattr(skill_module, 'pick', None),
        'place': getattr(skill_module, 'place', None),
        'move': getattr(skill_module, 'move', None),
        'push': getattr(skill_module, 'push', None),
        'open_gripper': getattr(skill_module, 'open_gripper', None),
        'close_gripper': getattr(skill_module, 'close_gripper', None),
        'align_two_axes': getattr(skill_module, 'align_two_axes', None),
        'align_to_quaternion': getattr(skill_module, 'align_to_quaternion', None),
        'normalize_quaternion': getattr(skill_module, 'normalize_quaternion', None),
        'angle_diff': getattr(skill_module, 'angle_diff', None),

        'ur5_move_to': getattr(skill_module, 'ur5_move_to', None),
        'ur5_grasp_at': getattr(skill_module, 'ur5_grasp_at', None),
        'ur5_release_at': getattr(skill_module, 'ur5_release_at', None),
        'ur5_align_gripper': getattr(skill_module, 'ur5_align_gripper', None),
        'close_ur5_ee': getattr(skill_module, 'close_ur5_ee', None),
        'open_ur5_ee': getattr(skill_module, 'open_ur5_ee', None),

        'sawyer_move_to': getattr(skill_module, 'sawyer_move_to', None),
        'sawyer_align_gripper': getattr(skill_module, 'sawyer_align_gripper', None),
        'sawyer_open_gripper': getattr(skill_module, 'sawyer_open_gripper', None),
        'sawyer_close_gripper': getattr(skill_module, 'sawyer_close_gripper', None),
        'sawyer_pick': getattr(skill_module, 'sawyer_pick', None),
        'sawyer_place': getattr(skill_module, 'sawyer_place', None),
    }


    import math
    import time
    context['math'] = math
    context['time'] = time
    context['sleep'] = time.sleep

    return context






def _count_run_skill_body_len(code_str: str) -> int:
    try:
        t = ast.parse(code_str)
    except Exception:
        return 0
    fn = None
    for n in getattr(t, "body", []):
        if isinstance(n, ast.FunctionDef) and n.name == "run_skill":
            fn = n
            break
    return len(fn.body) if fn and fn.body else 0

def debug_next_index_and_statements(
    current_code: str,
    current_lines: list[str],
    executed_stmt_texts: list[str],
    k: int = 8,
    tag: str = "DBG",
):
    try:
        tree = ast.parse(current_code)
    except Exception as e:
        print(f"[{tag}] current_code unparsable: {e}")
        return

    extractor = RunSkillExtractor()
    extractor.visit(tree)

    body = extractor.run_skill_body or []
    next_index = len(executed_stmt_texts)

    print(f"[{tag}] next_index={next_index}, body_len={len(body)}, remaining={max(0, len(body)-next_index)}")

    if not body:
        print(f"[{tag}] run_skill_body is empty")
        return


    if next_index > len(body):
        print(f"[{tag}] WARNING: next_index({next_index}) > body_len({len(body)})")

    remaining = body[next_index:]
    for j, st in enumerate(remaining[:k], start=next_index + 1):
        kind = type(st).__name__
        try:
            code = get_statement_code(st, current_lines).rstrip()
            head = code.splitlines()[0] if code else ""
        except Exception as e:
            head = f"<failed to render: {e}>"


        ln = getattr(st, "lineno", None)
        eln = getattr(st, "end_lineno", None)
        print(f"[{tag}] to_exec[{j}] {kind} (lineno={ln}, end={eln}): {head}")

    if len(remaining) > k:
        print(f"[{tag}] ... ({len(remaining) - k} more statements)")


def execute_failure_loop(
    env, task, task_name: str, robot_type: str,
    source_code: str, source_lines: List[str],
    max_skill_calls: Optional[int] = None,
    provider: Optional["PolicyProvider"] = None,
    max_repair_attempts: int = 20
) -> Optional[str]:

    """
    Execute policy in 'failure' mode.

    Returns:
        Final code after all repairs and revisions (or None on early error)
    """
    current_code = source_code

    _, objects_info, _ = get_object_names(task, task_name=task_name)
    type_map = {obj["name"]: obj["type"] for obj in objects_info}
    current_code = fix_type_mismatches(current_code, type_map)

    current_lines = current_code.split('\n')
    executed_stmt_texts: List[str] = []
    skill_call_count = 0
    repair_attempt_count = 0

    context = create_execution_context(env, task, robot_type)


    context['descriptions'] = None
    context['obs'] = task.get_observation()
    context['variations_index'] = 0
    context['variation_index'] = 0


    while True:
        try:
            tree = ast.parse(current_code)
        except SyntaxError as syn_err:

            print(f"[FAILURE] SyntaxError: {syn_err}")
            scene_info = get_scene_info(task, task_name, robot_type)
            failure_payload = {
                "type": "failure",
                "step_index": len(executed_stmt_texts),
                "statement": str(syn_err.text).strip() if syn_err.text else "",
                "error": str(syn_err),
                "error_type": "SyntaxError",
                "scene": scene_info
            }
            output_json(failure_payload)

            if provider is not None:
                repair_attempt_count += 1
                if repair_attempt_count > max_repair_attempts:
                    print(f"[MAX_REPAIRS] Reached maximum repair attempts: {max_repair_attempts}")
                    output_json({
                        "type": "max_repairs_reached",
                        "repair_attempt_count": repair_attempt_count,
                        "max_repair_attempts": max_repair_attempts,
                        "last_error": str(syn_err),
                        "last_error_type": "SyntaxError",
                        "success": False,
                        "scene": scene_info
                    })
                    return current_code

                try:

                    current_code = remove_truncated_lines(current_code)

                    exploration_obs = compute_static_observation(
                        task_name=task_name,
                        scene_info=scene_info,
                        failed_stmt_text=str(syn_err.text).strip() if syn_err.text else "",
                    )
                    new_code = provider.repair_on_failure(
                        current_code=current_code,
                        executed_stmt_texts=executed_stmt_texts,
                        failed_stmt_text=str(syn_err.text).strip() if syn_err.text else "",
                        error_payload={"error": str(syn_err), "error_type": "SyntaxError"},
                        scene=scene_info,
                        exploration_obs=exploration_obs,
                    )
                    print(f"[REPAIR] Code repaired after SyntaxError (attempt {repair_attempt_count}/{max_repair_attempts})")
                    cont_body = postprocess_continuation_code(new_code)

                    merged = merge_continuation_into_run_skill(
                        current_code, cont_body, insert_before_last_return=True,
                    )
                    merged = fix_type_mismatches(merged, type_map)
                    merged = remove_truncated_lines(merged)
                    current_code = merged
                    current_lines = current_code.split('\n')
                    continue
                except Exception as repair_error:
                    print(f"[REPAIR_FAILED] {repair_error}")
                    return current_code
            else:
                return current_code

        extractor = RunSkillExtractor()
        extractor.visit(tree)

        if not extractor.run_skill_body:
            print("[ERROR] run_skill 함수를 찾을 수 없습니다.")
            return current_code


        for imp_node in extractor.imports:
            try:
                imp_code = compile(ast.Module(body=[imp_node], type_ignores=[]), '<string>', 'exec')
                exec(imp_code, context)
            except Exception as imp_err:
                print(f"[WARN] Failed to execute import: {imp_err}")

        for _, func_node in extractor.helper_functions.items():
            func_code = compile(ast.Module(body=[func_node], type_ignores=[]), '<string>', 'exec')
            exec(func_code, context)

        next_index = len(executed_stmt_texts)
        statements_to_execute = extractor.run_skill_body[next_index:]

        if not statements_to_execute:
            success_check, _ = task._task.success()
            scene_info = get_scene_info(task, task_name, robot_type)

            if success_check:
                output_json({
                    "type": "complete",
                    "total_steps": len(executed_stmt_texts),
                    "total_skill_calls": skill_call_count,
                    "success": True,
                    "scene": scene_info
                })
                return current_code

            if max_skill_calls != None and skill_call_count >= max_skill_calls:
                print(f"[MAX_STEPS] Reached maximum skill calls: {max_skill_calls}")
                output_json({
                    "type": "max_steps_reached",
                    "total_steps": len(executed_stmt_texts),
                    "skill_call_count": skill_call_count,
                    "max_skill_calls": max_skill_calls,
                    "success": False,
                    "scene": scene_info
                })
                return current_code


            print(f"[INFO] All statements executed but task incomplete. Requesting additional code...")
            if provider != None:
                repair_attempt_count += 1
                if repair_attempt_count > max_repair_attempts:
                    print(f"[MAX_REPAIRS] Reached maximum repair/revise attempts: {max_repair_attempts}")
                    output_json({
                        "type": "max_repairs_reached",
                        "step_index": len(executed_stmt_texts),
                        "repair_attempt_count": repair_attempt_count,
                        "max_repair_attempts": max_repair_attempts,
                        "last_error": "Task incomplete after all statements executed",
                        "last_error_type": "RevisionLimitReached",
                        "success": False,
                        "scene": scene_info
                    })
                    return current_code

                last_step_payload = {
                    "step_index": len(executed_stmt_texts),
                    "skill_call_count": skill_call_count,
                    "statement": executed_stmt_texts[-1] if executed_stmt_texts else "",
                    "skill_name": None,
                    "success": False,
                    "need_more_steps": True,
                }


                exploration_obs = compute_static_observation(
                    task_name=task_name,
                    scene_info=scene_info,
                    failed_stmt_text=executed_stmt_texts[-1] if executed_stmt_texts else "",
                )

                if exploration_obs:
                    print(f"[EXPLORATION] Observation: {json.dumps(exploration_obs, indent=2)}")

                new_code = provider.repair_on_not_finished(
                    current_code=current_code,
                    executed_stmt_texts=executed_stmt_texts,
                    next_index=len(executed_stmt_texts),
                    scene=scene_info,
                    last_step_payload=last_step_payload,
                    exploration_obs=exploration_obs,
                )


                hyp_prompt = getattr(provider, 'last_hypothesis_prompt', None)
                hyp_response = getattr(provider, 'last_hypothesis_response', None)
                if hyp_prompt or hyp_response:
                    print(f"\n[HYPOTHESIS] === LLM Hypothesis Generation (revise) ===")
                    if hyp_prompt:
                        hyp_prompt_short = hyp_prompt[-500:] if len(hyp_prompt) > 500 else hyp_prompt

                    if hyp_response:
                        print(f"[HYPOTHESIS] LLM Response:\n{hyp_response}")
                    print(f"[HYPOTHESIS] ==========================================\n")

                if new_code:
                    print(f"[REVISE] Additional code generated (attempt {repair_attempt_count}/{max_repair_attempts})")

                    cont_body = postprocess_continuation_code(new_code)



                    merged = merge_continuation_into_run_skill(
                        current_code, cont_body, insert_before_last_return=True, insert_after_lineno=None)
                    merged = fix_type_mismatches(merged, type_map)
                    merged = remove_truncated_lines(merged)
                    current_code = merged

                    current_lines = current_code.split('\n')

                    if executed_stmt_texts and executed_stmt_texts[-1].lstrip().startswith("return"):
                        executed_stmt_texts.pop()
                    debug_next_index_and_statements(
                        current_code=current_code,
                        current_lines=current_lines,
                        executed_stmt_texts=executed_stmt_texts,
                        k=3,
                        tag="REVISE_AFTER_MERGE"
                    )
                    continue

            output_json({
                "type": "complete",
                "total_steps": len(executed_stmt_texts),
                "total_skill_calls": skill_call_count,
                "success": False,
                "scene": scene_info
            })
            return current_code

        repair_needed = False

        for statement_index, stmt in enumerate(statements_to_execute, start=next_index + 1):
            stmt_code = get_statement_code(stmt, current_lines)
            is_skill, skill_name = is_skill_call(stmt)


            if isinstance(stmt, ast.Return):
                print(f"[STEP {statement_index}] Skipping return statement (end of run_skill)")
                executed_stmt_texts.append(stmt_code)
                next_index += 1

                break


            if contains_return(stmt):
                print(f"[STEP {statement_index}] Removing return statements from: {stmt_code[:80]}...")
                stmt = remove_returns_from_stmt(stmt)
                ast.fix_missing_locations(stmt)

            try:
                stmt_module = ast.Module(body=[stmt], type_ignores=[])
                ast.fix_missing_locations(stmt_module)
                compiled = compile(stmt_module, '<string>', 'exec')
                exec(compiled, context)

                executed_stmt_texts.append(stmt_code)
                next_index += 1

                if is_skill:
                    skill_call_count += 1
                    success_check, _ = task._task.success()
                    if success_check:
                        scene_info = get_scene_info(task, task_name, robot_type)
                        output_json({
                            "type": "complete",
                            "step_index": statement_index,
                            "skill_call_count": skill_call_count,
                            "success": True,
                            "scene": scene_info
                        })
                        print(f"[SUCCESS] Task completed at step {statement_index}")
                        return current_code

                    if max_skill_calls != None and skill_call_count >= max_skill_calls:
                        scene_info = get_scene_info(task, task_name, robot_type)
                        print(f"[MAX_STEPS] Reached maximum skill calls: {max_skill_calls}")
                        output_json({
                            "type": "max_steps_reached",
                            "step_index": statement_index,
                            "skill_call_count": skill_call_count,
                            "max_skill_calls": max_skill_calls,
                            "success": success_check,
                            "scene": scene_info
                        })
                        repair_needed = True
                        break


            except BaseException as e:
                error_msg = getattr(e, 'message', str(e))
                scene_info = get_scene_info(task, task_name, robot_type)


                alignment_guidance = None
                gripper_quat = scene_info.get("gripper", {}).get("quaternion")
                if gripper_quat is not None:
                    alignment_guidance = check_gripper_alignment_for_scene(
                        np.array(gripper_quat),
                        scene_info.get("objects", []),
                        task_name,
                        robot_type
                    )

                failure_payload = {
                    "type": "failure",
                    "step_index": statement_index,
                    "statement": stmt_code,
                    "skill_name": skill_name if is_skill else None,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "scene": scene_info
                }
                if alignment_guidance:
                    failure_payload["alignment_guidance"] = alignment_guidance

                output_json(failure_payload)
                print(f"[FAILURE] {type(e).__name__}: {error_msg}")
                if alignment_guidance:
                    print(f"[ALIGNMENT] {alignment_guidance}")

                if provider != None:
                    repair_attempt_count += 1
                    if repair_attempt_count > max_repair_attempts:
                        print(f"[MAX_REPAIRS] Reached maximum repair attempts: {max_repair_attempts}")
                        output_json({
                            "type": "max_repairs_reached",
                            "step_index": statement_index,
                            "repair_attempt_count": repair_attempt_count,
                            "max_repair_attempts": max_repair_attempts,
                            "last_error": error_msg,
                            "last_error_type": type(e).__name__,
                            "success": False,
                            "scene": scene_info
                        })
                        return current_code

                    try:

                        exploration_obs = compute_static_observation(
                            task_name=task_name,
                            scene_info=scene_info,
                            failed_stmt_text=stmt_code,
                        )

                        if exploration_obs:
                            print(f"[EXPLORATION] Observation: {json.dumps(exploration_obs, indent=2)}")

                        new_code = provider.repair_on_failure(
                            current_code=current_code,
                            executed_stmt_texts=executed_stmt_texts,
                            failed_stmt_text=stmt_code,
                            error_payload={"error": error_msg, "error_type": type(e).__name__},
                            scene=scene_info,
                            exploration_obs=exploration_obs,
                        )


                        hyp_prompt = getattr(provider, 'last_hypothesis_prompt', None)
                        hyp_response = getattr(provider, 'last_hypothesis_response', None)
                        if hyp_prompt or hyp_response:
                            print(f"\n[HYPOTHESIS] === LLM Hypothesis Generation (repair) ===")
                            if hyp_prompt:
                                hyp_prompt_short = hyp_prompt[-500:] if len(hyp_prompt) > 500 else hyp_prompt

                            if hyp_response:
                                print(f"[HYPOTHESIS] LLM Response:\n{hyp_response}")
                            print(f"[HYPOTHESIS] ==========================================\n")

                        print(f"[REPAIR] Code repaired after failure (attempt {repair_attempt_count}/{max_repair_attempts})")


                        cont_body = postprocess_continuation_code(new_code)


                        after_ln_0based = int(getattr(stmt, "end_lineno", stmt.lineno)) - 1
                        merged = merge_continuation_into_run_skill(
                            current_code, cont_body, insert_before_last_return=False, insert_after_lineno=after_ln_0based,
                        )
                        merged = fix_type_mismatches(merged, type_map)
                        merged = remove_truncated_lines(merged)
                        current_code = merged

                        current_lines = current_code.split('\n')
                        repair_needed = True
                        break
                    
                    except Exception as repair_error:
                        print(f"[REPAIR_FAILED] {repair_error}")
                        return current_code
                else:
                    return current_code

        if not repair_needed:
            success_check, _ = task._task.success()
            scene_info = get_scene_info(task, task_name, robot_type)

            if success_check:
                output_json({
                    "type": "complete",
                    "total_steps": len(executed_stmt_texts),
                    "total_skill_calls": skill_call_count,
                    "success": True,
                    "scene": scene_info
                })
                return current_code

            if max_skill_calls != None and skill_call_count >= max_skill_calls:
                print(f"[MAX_STEPS] Reached maximum skill calls: {max_skill_calls}")
                output_json({
                    "type": "max_steps_reached",
                    "total_steps": len(executed_stmt_texts),
                    "skill_call_count": skill_call_count,
                    "max_skill_calls": max_skill_calls,
                    "success": False,
                    "scene": scene_info
                })
                return current_code



            print(f"[INFO] All statements executed but task incomplete. Requesting additional code...")
            if provider != None:
                repair_attempt_count += 1
                if repair_attempt_count > max_repair_attempts:
                    print(f"[MAX_REPAIRS] Reached maximum repair/revise attempts: {max_repair_attempts}")
                    output_json({
                        "type": "max_repairs_reached",
                        "step_index": len(executed_stmt_texts),
                        "repair_attempt_count": repair_attempt_count,
                        "max_repair_attempts": max_repair_attempts,
                        "last_error": "Task incomplete after all statements executed",
                        "last_error_type": "RevisionLimitReached",
                        "success": False,
                        "scene": scene_info
                    })
                    return current_code

                last_step_payload = {
                    "step_index": len(executed_stmt_texts),
                    "skill_call_count": skill_call_count,
                    "statement": executed_stmt_texts[-1] if executed_stmt_texts else "",
                    "skill_name": None,
                    "success": False,
                    "need_more_steps": True,
                }


                exploration_obs = compute_static_observation(
                    task_name=task_name,
                    scene_info=scene_info,
                    failed_stmt_text=executed_stmt_texts[-1] if executed_stmt_texts else "",
                )

                if exploration_obs:
                    print(f"[EXPLORATION] Observation: {json.dumps(exploration_obs, indent=2)}")

                new_code = provider.repair_on_not_finished(
                    current_code=current_code,
                    executed_stmt_texts=executed_stmt_texts,
                    next_index=len(executed_stmt_texts),
                    scene=scene_info,
                    last_step_payload=last_step_payload,
                    exploration_obs=exploration_obs,
                )


                hyp_prompt = getattr(provider, 'last_hypothesis_prompt', None)
                hyp_response = getattr(provider, 'last_hypothesis_response', None)
                if hyp_prompt or hyp_response:
                    print(f"\n[HYPOTHESIS] === LLM Hypothesis Generation (revise) ===")
                    if hyp_prompt:
                        hyp_prompt_short = hyp_prompt[-500:] if len(hyp_prompt) > 500 else hyp_prompt

                    if hyp_response:
                        print(f"[HYPOTHESIS] LLM Response:\n{hyp_response}")
                    print(f"[HYPOTHESIS] ==========================================\n")

                if new_code:
                    print(f"[REVISE] Additional code generated (attempt {repair_attempt_count}/{max_repair_attempts})")

                    cont_body = postprocess_continuation_code(new_code)



                    merged = merge_continuation_into_run_skill(
                        current_code, cont_body, insert_before_last_return=True, insert_after_lineno=None)
                    merged = fix_type_mismatches(merged, type_map)
                    merged = remove_truncated_lines(merged)
                    current_code = merged

                    current_lines = current_code.split('\n')

                    if executed_stmt_texts and executed_stmt_texts[-1].lstrip().startswith("return"):
                        executed_stmt_texts.pop()
                    debug_next_index_and_statements(
                        current_code=current_code,
                        current_lines=current_lines,
                        executed_stmt_texts=executed_stmt_texts,
                        k=3,
                        tag="REVISE_AFTER_MERGE"
                    )
                    continue

            output_json({
                "type": "complete",
                "total_steps": len(executed_stmt_texts),
                "total_skill_calls": skill_call_count,
                "success": False,
                "scene": scene_info
            })
            return current_code








class EpisodeResult:
    """에피소드 실행 결과를 담는 클래스"""
    def __init__(self):
        self.success: bool = False
        self.execution_time: float = 0.0
        self.skill_call_count: int = 0
        self.total_steps: int = 0
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None
        self.llm_calls: List[Dict[str, Any]] = []
        self.final_code: Optional[str] = None
        self.prompts: List[Dict[str, str]] = []

        self.exception_log: List[Dict[str, Any]] = []
        self.termination_reason: Optional[str] = None
        self.traceback: Optional[str] = None


def run_task_episode(
    task_name: str,
    mode: str = "step",
    source_robot: str = "panda",
    target_robot: str = "panda",
    max_skill_calls: int = 10,
    max_repair_attempts: int = 20,
    code_source: str = "remote_llm",
    model: str = "ours",
    remote_host: str = "172.17.0.1",
    remote_port: int = 5000,
    llm_call_logger: Optional[callable] = None,
    seed: int | None = None,
    variation: int = 0,
) -> EpisodeResult:
    """
    단일 에피소드를 실행하는 함수.
    main()과 동일한 로직이지만, argument를 함수 파라미터로 받음.

    Args:
        task_name: RLBench task class name (예: BasketballInHoop)
        mode: 실행 모드 (step 또는 failure)
        source_robot: Source robot type (panda, ur5, sawyer, jaco)
        target_robot: Target robot type for environment setup
        max_skill_calls: Maximum number of primitive skill calls (None = unlimited)
        max_repair_attempts: Maximum number of repair attempts before giving up (default: 5)
        code_source: Where to get task skill code (static, local_llm, remote_llm)
        model: Model type (code_agent, ours)
        remote_host: Remote LLM server host
        remote_port: Remote LLM server port
        llm_call_logger: Optional callback to log LLM calls (called with dict: action, duration, etc.)

    Returns:
        EpisodeResult: 에피소드 실행 결과
    """
    import time as time_module

    result = EpisodeResult()
    start_time = time_module.time()

    source_robot = source_robot.lower()
    target_robot = target_robot.lower()
    code_source = code_source.lower()
    model = model.lower()

    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Source Robot: {source_robot}")
    print(f"[INFO] Target Robot: {target_robot}")
    print(f"[INFO] Max Skill Calls: {max_skill_calls if max_skill_calls else 'unlimited'}")


    tasks_module = importlib.import_module("rlbench.tasks")
    try:
        task_cls = getattr(tasks_module, task_name)
    except AttributeError:
        print(f"[ERROR] '{task_name}' 태스크 클래스를 찾을 수 없습니다.")
        result.error = f"Task class '{task_name}' not found"
        result.error_type = "TaskNotFound"
        result.execution_time = time_module.time() - start_time
        return result


    env, task = None, None
    provider = None

    try:
        print(f"[INFO] Setting up environment with {target_robot} robot...")
        env, task = setup_environment(task_cls, target_robot, seed=seed, variation=variation)


        descriptions, _ = task.reset()


        initial_info = get_initial_info(task, task_name, target_robot, descriptions)


        output_json(initial_info)


        import json, hashlib

        scene_obj = initial_info.get("scene", {})


        scene_str = json.dumps(scene_obj, sort_keys=True, default=str)
        scene_md5 = hashlib.md5(scene_str.encode("utf-8")).hexdigest()[:8]





        policy_file_path = f"./tasks_{source_robot}/{task_name}.py"

        objects_info_str, objects_info, object_names = get_object_names(task, task_name)
        grasp_guidance = initial_info.get("scene", {}).get("grasp_guidance", "")
        initial_prompt = get_prompt(task, task_name, object_names, target_robot, descriptions, model, grasp_guidance)













        provider = PolicyProvider(
            mode=code_source,
            policy_path=policy_file_path,
            task_name=task_name,
            source_robot=source_robot,
            target_robot=target_robot,
            model=model,
            initial_prompt=initial_prompt,
            remote_host=remote_host,
            remote_port=remote_port,
            target_scene_info=objects_info,
            grasp_guidance=grasp_guidance,
        )


        provider = _wrap_provider_with_timing(provider, task_name, llm_call_logger, result)

        print(f"[INFO] PolicyProvider initialized with mode: {code_source}")


        try:
            source_code = provider.get_initial_code()


            if code_source != 'static':
                source_code = postprocess_initial_code(source_code, target_robot)




            source_lines = source_code.split('\n')
            print(f"[INFO] Loaded policy from: {policy_file_path}")
        except Exception as e:
            print(f"[ERROR] Failed to get initial code: {e}")
            result.error = str(e)
            result.error_type = type(e).__name__
            result.execution_time = time_module.time() - start_time
            return result

        print(source_code)


        final_code = None
        final_code = execute_failure_loop(env, task, task_name, target_robot, source_code, source_lines, max_skill_calls, provider, max_repair_attempts)


        result.final_code = final_code if final_code else source_code


        success_check, _ = task._task.success()
        result.success = success_check


        if success_check:
            result.termination_reason = "success"
        else:
            result.termination_reason = "task_failure"

    except Exception as e:
        print(f"[ERROR] Episode execution failed: {e}")
        tb = traceback.format_exc()
        traceback.print_exc()
        result.error = str(e)
        result.error_type = type(e).__name__
        result.traceback = tb
        result.termination_reason = "exception"
        result.exception_log.append({
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb,
            "phase": "episode_execution"
        })
        result.success = False
    finally:

        if provider != None:

            actual_provider = getattr(provider, '_wrapped_provider', provider)
            actual_provider.close()


        if env != None:
            print("[INFO] Shutting down environment...")
            shutdown_environment(env)
            print("[INFO] Done.")

        result.execution_time = time_module.time() - start_time

    return result



class _TimingProviderWrapper:
    """PolicyProvider를 감싸서 LLM 호출 시간을 측정하는 래퍼"""

    def __init__(self, provider: PolicyProvider, task_name: str, logger: callable, result: EpisodeResult):
        self._wrapped_provider = provider
        self._task_name = task_name
        self._logger = logger
        self._result = result

    def __getattr__(self, name):
        return getattr(self._wrapped_provider, name)

    def _log_call(self, action: str, subtask: str, duration: float, num_tokens: Optional[int] = None):
        call_info = {
            "task": self._task_name,
            "subtask": subtask,
            "action": action,
            "duration_sec": duration,
        }
        if num_tokens is not None:
            call_info["num_tokens"] = num_tokens
        self._result.llm_calls.append(call_info)
        if self._logger:
            self._logger(call_info)

        last_prompt = getattr(self._wrapped_provider, 'last_prompt', None)
        if last_prompt:
            prompt_entry = {
                "action": action,
                "subtask": subtask,
                "prompt": last_prompt,
            }

            hyp_prompt = getattr(self._wrapped_provider, 'last_hypothesis_prompt', None)
            hyp_response = getattr(self._wrapped_provider, 'last_hypothesis_response', None)
            if hyp_prompt:
                prompt_entry["hypothesis_prompt"] = hyp_prompt
            if hyp_response:
                prompt_entry["hypothesis_response"] = hyp_response
            self._result.prompts.append(prompt_entry)

    def get_initial_code(self) -> str:
        import time as t
        start = t.time()
        code = self._wrapped_provider.get_initial_code()
        duration = t.time() - start
        num_tokens = getattr(self._wrapped_provider, 'last_response_num_tokens', None)
        self._log_call("generate", "initial_code", duration, num_tokens)
        return code

    def revise_on_step(self, current_code, executed_stmt_texts, next_index, scene, last_step_payload) -> Optional[str]:
        import time as t
        start = t.time()
        code = self._wrapped_provider.revise_on_step(current_code, executed_stmt_texts, next_index, scene, last_step_payload)
        duration = t.time() - start
        num_tokens = getattr(self._wrapped_provider, 'last_response_num_tokens', None)
        step_idx = last_step_payload.get("step_index", next_index)
        self._log_call("revise", f"step_{step_idx}", duration, num_tokens)
        return code

    def repair_on_failure(self, current_code, executed_stmt_texts, failed_stmt_text, error_payload, scene, exploration_obs=None) -> str:
        import time as t
        start = t.time()
        code = self._wrapped_provider.repair_on_failure(current_code, executed_stmt_texts, failed_stmt_text, error_payload, scene, exploration_obs)
        duration = t.time() - start
        num_tokens = getattr(self._wrapped_provider, 'last_response_num_tokens', None)
        self._log_call("repair", f"failure_{error_payload.get('error_type', 'unknown')}", duration, num_tokens)
        return code

    def repair_on_not_finished(self, current_code, executed_stmt_texts, next_index, scene, last_step_payload, exploration_obs=None) -> str:
        import time as t
        start = t.time()
        code = self._wrapped_provider.repair_on_not_finished(current_code, executed_stmt_texts, next_index, scene, last_step_payload, exploration_obs)
        duration = t.time() - start
        num_tokens = getattr(self._wrapped_provider, 'last_response_num_tokens', None)
        step_idx = last_step_payload.get("step_index", next_index)
        self._log_call("revise", f"step_{step_idx}", duration, num_tokens)
        return code


def _wrap_provider_with_timing(provider: PolicyProvider, task_name: str, logger: callable, result: EpisodeResult):
    """PolicyProvider를 시간 측정 래퍼로 감쌈"""
    return _TimingProviderWrapper(provider, task_name, logger, result)






def main():
    parser = argparse.ArgumentParser(
        description="RLBench Loop - AST 기반 step-by-step 또는 failure-only 실행"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="RLBench task class name (예: BasketballInHoop, PushButton)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["step", "failure"],
        default="step",
        help="실행 모드: step(각 statement마다 출력) 또는 failure(실패 시에만 출력)"
    )
    parser.add_argument(
        "--source_robot", "-s",
        default="panda",
        help="Source robot type (panda, ur5, sawyer, jaco)"
    )
    parser.add_argument(
        "--target_robot", "-r",
        default="ur5",
        help="Target robot type for environment setup (panda, ur5, sawyer, jaco)"
    )
    parser.add_argument(
        "--max_steps", "-n",
        type=int,
        default=None,
        help="Maximum number of primitive skill calls (default: unlimited)"
    )
    parser.add_argument(
        "--code_source", "-c",
        type=str,
        default="remote_llm",
        choices=["static", "local_llm", "remote_llm"],
        help="Where to get task skill code: static | local_llm | remote_llm",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="code_agent",
        choices=["code_agent", "ours"],
    )
    parser.add_argument(
        "--host_ip",
        type=str,
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
    )
    parser.add_argument(
        "--max_repairs",
        type=int,
        default=20,
        help="Maximum number of repair attempts before giving up (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (applied to numpy/random in environment setup)"
    )
    parser.add_argument(
        "--variation",
        type=int,
        default=0,
        help="RLBench task variation index"
    )

    args = parser.parse_args()

    task_name = args.task
    mode = args.mode
    source_robot = args.source_robot.lower()
    target_robot = args.target_robot.lower()
    max_skill_calls = args.max_steps
    max_repair_attempts = args.max_repairs
    code_source = args.code_source.lower()
    model = args.model.lower()

    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Source Robot: {source_robot}")
    print(f"[INFO] Target Robot: {target_robot}")
    print(f"[INFO] Max Skill Calls: {max_skill_calls if max_skill_calls else 'unlimited'}")


    tasks_module = importlib.import_module("rlbench.tasks")
    try:
        task_cls = getattr(tasks_module, task_name)
    except AttributeError:
        print(f"[ERROR] '{task_name}' 태스크 클래스를 찾을 수 없습니다.")
        sys.exit(1)


    print(f"[INFO] Setting up environment with {target_robot} robot...")
    env, task = setup_environment(task_cls, target_robot)


    descriptions, _ = task.reset()


    initial_info = get_initial_info(task, task_name, target_robot, descriptions)


    output_json(initial_info)

    policy_file_path = f"./tasks_{source_robot}/{task_name}.py"

    objects_info_str, objects_info, object_names = get_object_names(task, task_name)
    grasp_guidance = initial_info.get("scene", {}).get("grasp_guidance", "")
    initial_prompt = get_prompt(task, task_name, object_names, target_robot, descriptions, model, grasp_guidance)



    _, objects_info = object_names if isinstance(object_names, tuple) else ("", object_names)
    object_names_list = objects_info_str


    provider = PolicyProvider(
        mode=code_source,
        policy_path=policy_file_path,
        task_name=task_name,
        source_robot=source_robot,
        target_robot=target_robot,
        model=model,
        initial_prompt=initial_prompt,
        remote_host=args.host_ip,
        remote_port=args.port,
        object_names=object_names_list,
        descriptions=descriptions,
        target_scene_info=objects_info,
        grasp_guidance=grasp_guidance,
    )

    print(f"[INFO] PolicyProvider initialized with mode: {code_source}")


    try:
        source_code = provider.get_initial_code()


        if code_source != 'static':
            source_code = postprocess_initial_code(source_code, target_robot)

        source_lines = source_code.split('\n')
        print(f"[INFO] Loaded policy from: {policy_file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to get initial code: {e}")
        provider.close()
        shutdown_environment(env)
        sys.exit(1)

    print(source_code)

    try:





        execute_failure_loop(env, task, task_name, target_robot, source_code, source_lines, max_skill_calls, provider, max_repair_attempts)
    finally:

        provider.close()

        print("[INFO] Shutting down environment...")
        shutdown_environment(env)
        print("[INFO] Done.")


if __name__ == "__main__":
    main()