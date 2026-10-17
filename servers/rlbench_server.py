# Host/Port: 127.0.0.1:9000
# Model: Qwen2.5-Coder-7B (base) or Qwen2.5-Coder-7B-Instruct (instruct)
# Usage:
#   python server.py --model-type base      # Use Qwen2.5-Coder-7B
#   python server.py --model-type instruct  # Use Qwen2.5-Coder-7B-Instruct
#   python server.py --model-dir /custom/path  # Use custom model path

import socket
import struct
import json
import threading
import argparse
import time
import re
import ast
from typing import Any, Dict, List, Optional, Tuple

import os

# -------------------------
# JSON framing
# -------------------------

def recvall(sock, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("client disconnected")
        data += chunk
    return data

def recv_json(sock) -> Dict[str, Any]:
    header = recvall(sock, 4)
    (length,) = struct.unpack(">I", header)
    payload = recvall(sock, length)
    return json.loads(payload.decode("utf-8"))

def send_json(sock, obj: Dict[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)

# -------------------------
# Prompt builders
# -------------------------

# reference code 주석 처리 (body-only, def 줄 제외, dedent 후 # 붙이기)
def _comment_out_run_skill_body(reference_code: str) -> str:
    """
    Extract only the body of run_skill (excluding `def run_skill(...):` line),
    dedent to base level, and comment out each line with `# `.
    Relative indentation inside the body is preserved.
    """
    code = (reference_code or "").strip("\n")
    if not code.strip():
        return code

    lines = code.splitlines()

    # Find "def run_skill" line
    def_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def run_skill"):
            def_idx = i
            break

    # If not found, just comment out everything (body-only)
    if def_idx is None:
        body_lines = lines
    else:
        # Collect only the body lines after def (indented or blank)
        body_lines = []
        for j in range(def_idx + 1, len(lines)):
            ln = lines[j]
            # Stop at a new top-level (non-indented, non-blank) statement
            if ln.strip() != "" and not ln.startswith((" ", "\t")):
                break
            body_lines.append(ln)

    if not body_lines or not any(ln.strip() for ln in body_lines):
        return "# (empty body)\n"

    # Dedent: find minimum indentation of non-empty body lines
    non_empty = [ln for ln in body_lines if ln.strip()]
    min_indent = min(len(ln) - len(ln.lstrip()) for ln in non_empty)

    # Comment out with dedented indentation
    out = []
    for ln in body_lines:
        if not ln.strip():
            out.append("#")
        else:
            dedented = ln[min_indent:]  # remove base indent, keep relative
            out.append("# " + dedented.rstrip())
    return "\n".join(out).rstrip() + "\n"




def build_scene_description(
    scene: Dict[str, Any],
    gripper_state: Optional[Dict[str, Any]] = None,
    executed_stmt_texts: Optional[List[str]] = None,
) -> str:
    """
    scene 정보를 자연어 형태의 묘사로 변환합니다.

    Args:
        scene: get_scene_info()에서 반환된 scene 정보
            - objects: list of {name, type, position, quaternion}
            - gripper: {position, quaternion, is_open}
        gripper_state: 추가적인 gripper 상태 정보 (held_object 등)
        executed_stmt_texts: 이미 실행된 statement 리스트

    Returns:
        scene 묘사 문자열
    """
    desc_parts = []

    # Gripper 상태 묘사
    gripper = scene.get("gripper", {})
    gripper_pos = gripper.get("position")
    is_open = gripper.get("is_open", True)

    gripper_desc = "# Current gripper state:\n"
    if gripper_pos:
        gripper_desc += f"#   Position: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]\n"
    gripper_desc += f"#   Gripper is {'OPEN' if is_open else 'CLOSED'}\n"

    # held_object 정보 추가
    if gripper_state:
        held_obj = gripper_state.get("held_object")
        if held_obj:
            gripper_desc += f"#   Currently holding: '{held_obj}'\n"
        elif not is_open:
            gripper_desc += f"#   Gripper closed but not holding any object\n"

    desc_parts.append(gripper_desc)

    # Objects 상태 묘사
    objects = scene.get("objects", [])
    if objects:
        obj_desc = "# Current object positions:\n"
        for obj in objects:
            name = obj.get("name", "unknown")
            obj_type = obj.get("type", "unknown")
            pos = obj.get("position")
            if pos:
                obj_desc += f"#   {name} ({obj_type}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            else:
                obj_desc += f"#   {name} ({obj_type}): position unknown\n"
        desc_parts.append(obj_desc)

    # 실행된 statements 요약
    if executed_stmt_texts:
        exec_desc = f"# Already executed {len(executed_stmt_texts)} statement(s):\n"
        for i, stmt in enumerate(executed_stmt_texts[-5:], start=max(0, len(executed_stmt_texts)-5)+1):
            # 마지막 5개만 표시
            stmt_short = stmt.strip().replace('\n', ' ')[:80]
            exec_desc += f"#   [{i}] {stmt_short}{'...' if len(stmt.strip()) > 80 else ''}\n"
        if len(executed_stmt_texts) > 5:
            exec_desc = f"# ... ({len(executed_stmt_texts) - 5} earlier statements omitted)\n" + exec_desc.split('\n', 1)[1]
        desc_parts.append(exec_desc)

    return "\n".join(desc_parts)


def _build_hypothesis_prompt(
    base_prompt: str,
    parts_without_instruction: List[str],
    branch_type: str,
    object_names: Optional[List[str]] = None,
    skill_list: Optional[str] = None,
) -> str:
    """
    기존 revise/repair 프롬프트에서 '# Composition patterns', '# RULES (IMPORTANT)',
    Instruction 섹션을 제거한 hypothesis 생성용 프롬프트를 조립한다.

    Args:
        base_prompt: get_revise_prompt / get_repair_prompt 결과 (rules + skill shapes 포함)
        parts_without_instruction: reference code, scene, executed code 등 (Instruction 제외)
        branch_type: "revise" or "repair"
        object_names: scene에 존재하는 object 이름 목록
        skill_list: 사용 가능한 primitive skill 목록 문자열
    Returns:
        hypothesis 생성용 프롬프트 문자열
    """
    # base_prompt에서 ## Current state / # REVISE: or # REPAIR: 블록,
    # # Composition patterns 섹션, # RULES (IMPORTANT) 섹션 제거
    filtered_lines = []
    skip_current_state = False
    skip_composition = False
    skip_rules = False
    for line in base_prompt.splitlines():
        stripped = line.strip()
        # "## Current state" 시작 시 스킵 시작 (# REVISE: / # REPAIR: 블록 포함)
        if stripped == "## Current state":
            skip_current_state = True
            continue
        if skip_current_state:
            # '#' 로 시작하는 설명 줄은 스킵, 빈 줄도 스킵
            if stripped.startswith("#") or stripped == "":
                continue
            else:
                skip_current_state = False
        # "# Composition patterns" 시작 시 스킵 시작
        if stripped.startswith("# Composition patterns"):
            skip_composition = True
            continue
        # "# RULES (IMPORTANT)" 시작 시 스킵 시작
        if stripped == "# RULES (IMPORTANT)":
            skip_rules = True
            continue
        # Composition patterns 블록 끝: 빈 줄이 아닌 새 섹션 시작
        if skip_composition:
            # 주석 예제 블록이 끝나면 (비어있지 않은 줄이 # 로 시작하지 않거나, 새 섹션 마커)
            if stripped and not stripped.startswith("#"):
                skip_composition = False
            else:
                continue
        # RULES 블록 끝: 빈 줄 후 다른 섹션
        if skip_rules:
            if stripped.startswith("# -") or stripped == "":
                continue
            else:
                skip_rules = False
        filtered_lines.append(line)

    filtered_base = "\n".join(filtered_lines).strip()

    # parts에서 Instruction 관련 부분 제외 (이미 parts_without_instruction으로 전달됨)
    hyp_parts = [filtered_base]
    hyp_parts.extend(parts_without_instruction)

    # Available objects / skills → ## Current state 직전에 삽입
    if object_names:
        obj_list_str = ", ".join(f"'{n}'" for n in object_names)
        hyp_parts.append(f"\n# Available objects: {obj_list_str}")
    if skill_list:
        hyp_parts.append(f"# Available primitive skills: {skill_list}")

    # Hypothesis 생성 지시 (branch_type에 따라 다르게 구성)
    hyp_parts.append("\n## Current state")

    if branch_type == "repair":
        # repair: 에러가 발생한 상황 - 에러 원인과 수정 방법에 집중
        hyp_parts.append("# HYPOTHESIS: While in this state, execution failed with an error. Your objective is to form a concise, evidence-based hypothesis about (1) why the execution failed with the specific error and (2) what should change in the code to fix it (where/what). Focus on the error signal, failed statement, and observations. Do not propose concrete code yet—output only the hypothesis statements.")
    else:  # revise
        # revise: 작업이 완료되지 않은 상황 - 추가로 필요한 단계에 집중
        hyp_parts.append("# HYPOTHESIS: While in this state, all statements executed but the task goal was not achieved. Your objective is to form a concise, evidence-based hypothesis about (1) why the task remained incomplete and (2) what additional steps or changes are needed in the code (where/what). Use the provided observations and traces. Do not propose concrete code yet—output only the hypothesis statements.")

    hyp_parts.append("# Output ONLY the hypothesis as 2-3 commented lines starting with '# '.")
    hyp_parts.append("# HYPOTHESIS:")

    return "\n".join(hyp_parts).strip() + "\n"


def _generate_hypothesis_with_llm(engine: 'LLMEngine', hypothesis_prompt: str) -> str:
    """
    LLM을 호출하여 hypothesis를 생성한다.
    Returns:
        "# HYPOTHESIS:\n# - line1\n# - line2\n# - line3" 형태의 문자열
    """
    gen = {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "min_new_tokens": 20,
        "max_new_tokens": 150,  # hypothesis는 2-3줄이면 충분
    }

    if _DEBUG_RPC:
        print("\n[HYPOTHESIS_PROMPT]\n", hypothesis_prompt)

    out_text, new_tokens, dt = engine.generate_chat(hypothesis_prompt, gen)

    if _DEBUG_RPC:
        print(f"[HYPOTHESIS_RESULT] tokens={new_tokens}, dt={dt:.2f}s")
        print(out_text)

    # 생성된 텍스트에서 hypothesis 부분만 추출 (# 주석 줄만, 최대 4줄)
    hyp_lines = []
    for line in out_text.strip().splitlines():
        line = line.strip()
        if line.startswith("#"):
            hyp_lines.append(line)
            if len(hyp_lines) >= 4:
                break
        elif hyp_lines:
            # hypothesis 줄이 끝나면 중단
            break

    if not hyp_lines:
        return ""

    return "# HYPOTHESIS:\n" + "\n".join(hyp_lines)


def get_revise_prompt(
    task_name: str,
    target_robot: str,
    scene: Dict[str, Any],
    executed_stmt_texts: List[str],
    last_step_payload: Dict[str, Any],
    object_names: List[str],
) -> str:

    obj_list_str = ", ".join(f"'{n}'" for n in object_names) if object_names else "(none)"

    if (target_robot or "").lower() == "ur5":
        skill_list = "ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper, close_ur5_ee, open_ur5_ee"
        skill_examples = """
# Primitive call shapes (reference)
# ur5_grasp_at:  obs, reward, done = ur5_grasp_at(env, task, grasp_pos=[x,y,z], approach={{'axis':'z', 'distance':FLOAT}}, timeout_s=FLOAT)
# ur5_release_at: obs, reward, done = ur5_release_at(env, task, place_pos=[x,y,z], approach={{'axis':'z', 'distance':FLOAT}}, timeout_s=FLOAT)
# ur5_move_to:  obs, reward, done = ur5_move_to(env, task, target_pos=[x,y,z], timeout_s=FLOAT)  # rare; ur5_grasp_at/ur5_release_at usually enough
# ur5_align_gripper:
#   q = obj.get_quaternion()
#   obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction=APPROACH_DIR, timeout_s=FLOAT)
# close_ur5_ee: obs, reward, done = close_ur5_ee(env, task, gripper_close=0.0, velocity=0.2)  # close gripper (0.0=closed, 1.0=open)
# open_ur5_ee:  obs, reward, done = open_ur5_ee(env, task, gripper_open=1.0, velocity=0.2)   # open gripper (0.0=closed, 1.0=open)

# Composition patterns (reference)
# Example 1. ur5_grasp_at then ur5_release_at
# obj = Shape('object_name')
# pos = obj.get_position()
# target_pos = pos + [0.0, 0.0, 0.08]
# obs, reward, done = ur5_grasp_at(env, task, grasp_pos=pos, approach={{'axis':____, 'distance':____}}, timeout_s=____)
# obs, reward, done = ur5_release_at(env, task, place_pos=target_pos, approach={{'axis':____, 'distance':____}}, timeout_s=____)
#
# Example 2. ur5_move_to() then, ur5_align_gripper then, ur5_grasp_at/ur5_release_at
# above_grasp_pos = pos + [0.0, 0.0, 0.08]
# obs, reward, done = ur5_move_to(env, task, target_pos=above_grasp_pos, timeout_s=FLOAT)
# q = obj.get_quaternion()
# obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode=____, approach_direction=____, timeout_s=____)
"""
    elif (target_robot or "").lower() == "jaco":
        skill_list = "jaco_move_linear, jaco_move_arc, jaco_rotate_gripper, jaco_open_gripper, jaco_close_gripper, jaco_align_gripper"
        skill_examples = """
# Primitive call shapes (reference)

"""
    elif (target_robot or "").lower() == "sawyer":
        skill_list = "sawyer_move_to, sawyer_pick, sawyer_place, sawyer_align_gripper, sawyer_open_gripper, sawyer_close_gripper"
        skill_examples = """
# - sawyer_align_gripper is NOT A DEFAULT STEP. DO NOT USE it.

# Primitive call shapes (reference)
# sawyer_pick:  obs, reward, done = sawyer_pick(env, task, target_object=obj, target_pos=[x,y,z], grasp_offset=[dx,dy,dz])
# sawyer_place: obs, reward, done = sawyer_place(env, task, place_pos=[x,y,z], place_offset=[dx,dy,dz])
# sawyer_move_to: obs, reward, done = sawyer_move_to(env, task, target_pos=[x,y,z], timeout_s=FLOAT)
# sawyer_open_gripper:  obs, reward, done = sawyer_open_gripper(env, task, amount=1.0, velocity=0.2)
# sawyer_close_gripper: obs, reward, done = sawyer_close_gripper(env, task, amount=0.0, velocity=0.2)

# Utility functions available (reference):
# - get_bbox_sizes(obj): returns (sizes, bbox) where sizes=[dx,dy,dz]
# - quat_mul(q1, q2): quaternion multiplication
# - normalize_vector(v): normalize vector with error handling

# - STRICTLY FOLLOW target_pos=target_pos, grasp_offset=np.array([0.0, 0.0, 0.0]). STRICTLY FOLLOW Composition patterns (reference) below.

# Composition patterns (reference)
# Example 1. pick then lift (do not use sawyer_align_gripper)
# target = Shape('target_name'); target_pos = target.get_position()
# success = ProximitySensor('success'); success_pos = success.get_position()
# obs, reward, done = sawyer_pick(env, task, target_object=target, target_pos=target_pos, grasp_offset=np.array([0.0, 0.0, 0.0]))
# obs, reward, done = sawyer_move_to(env, task, target_pos=success_pos, timeout_s=10.0)
"""
    else:
        skill_list = "move, pick, place, push, open_gripper, close_gripper, align_to_quaternion, align_two_axes"
        skill_examples = """
# Primitive call shapes (reference)
# pick:  obs, reward, done = pick(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
# place: obs, reward, done = place(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
# move:  obs, reward, done = move(env, task, target_pos=[x,y,z], timeout=FLOAT)
# push:  obs, reward, done = push(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
# grip:  obs, reward, done = open_gripper(env, task) / close_gripper(env, task)
# align:
#   q = obj.get_quaternion()
#   obs, reward, done = align_to_quaternion(env, task, quaternion=q, yaw_align='parallel', approach_dir=APPROACH_DIR, timeout=FLOAT)
"""

    approach_description = """# - approach must be a DICT with 'axis' (STRING in {{'x','y','z','-x','-y','-z'}}) and 'distance' (FLOAT) (do NOT use AXIS.* or enums)."""

    control_flow_rule = "# - DO NOT use if done: / else branching or other control-flow constructs (e.g., while loops)."

    # Robot-specific rules
    if (target_robot or "").lower() == "ur5":
        robot_specific_rules = """# - ur5_grasp_at/ur5_release_at already include moving/approach internally. Do NOT add extra ur5_move_to() unless truly needed.
# - Unless there is a specific reason, use only 'z' or '-z' for approach['axis'] (avoid 'x'/'y' approaches) for ur5_grasp_at.
# - If "GRASP_GUIDANCE" is provided in the scene/error payload, implement it in code exactly
# - Before calling ur5_align_gripper, first move the gripper to a safe pre-grasp pose (e.g., above_grasp_pos) using ur5_move_to().
# - For push-related tasks, call close_ur5_ee() before any push/press motion.
# - ur5_align_gripper is NOT a default step. Use it ONLY when top-down approach is not possible.
# - yaw_mode MUST be either 'parallel' or 'perpendicular' (no other values allowed)"""
    elif (target_robot or "").lower() == "sawyer":
        robot_specific_rules = """# - sawyer_pick/sawyer_place already include approach/retract internally. Do NOT add extra sawyer_move_to() before pick/place unless truly needed.
# - sawyer_align_gripper is NOT a default step. DO NOT USE it unless specifically required.
# - For push-related tasks, call sawyer_close_gripper() before any push/press motion.
# - STRICTLY FOLLOW target_pos=target_pos, grasp_offset=np.array([0.0, 0.0, 0.0]) for sawyer_pick."""
    elif (target_robot or "").lower() == "jaco":
        robot_specific_rules = """# - jaco_move_linear is for straight-line motions; jaco_move_arc for curved paths.
# - Always call jaco_open_gripper/jaco_close_gripper explicitly before/after grasping."""
    else:
        robot_specific_rules = """# - If "GRASP_GUIDANCE" is provided in the scene/error payload, implement it in code exactly"""

    prompt = f"""# Task: {task_name}
# You MUST use the EXACT objects and skills provided. NO placeholders, NO new APIs.
#
## Current state
# REVISE: While in this state, the task is still not successful. The code executed so far is NOT sufficient (and may be wrong) because it did not achieve success. Your objective is to generate a corrected continuation from the CURRENT environment state: do NOT merely follow or extend the previously executed steps. Use the hypothesis, REFERENCE CODE, error signal, and observations to infer the most likely cause, then output ONLY a continuation patch to reach task success from the given resume point.
#
# Available objects: {obj_list_str}
# Available primitive skills: {skill_list}

# RULES (IMPORTANT)
{control_flow_rule}
# - Every primitive skill call MUST be written as: obs, reward, done = <skill>(...)
# - Never access env.scene or env.scene.objects (it does not exist).
{robot_specific_rules}
# - Use a longer timeout_s than 5.0 when needed (e.g., 8.0-15.0) to allow sufficient time for waypoint motion and avoid premature failures.
# - Use reasonable values: approach distance ~ 0.05-0.15, timeout_s ~ 8.0-15.0.
# - offset is applied when computing target_pos: target_pos = object_pos + [dx, dy, dz]
#   Example: target_pos = pos + [0.0, 0.0, 0.2]  # 20cm above
# - Output ONLY executable Python statements. ABSOLUTELY NO comments: never output '#' anywhere (even inline).
# - NEVER output any tag-like text (no angle brackets '<' or '>'), and never output the substring 'tool_call'. Only Python statements.

{skill_examples}
"""
    return prompt


def get_repair_prompt(
    task_name: str,
    target_robot: str,
    current_code: str,
    executed_stmt_texts: List[str],
    failed_stmt_text: str,
    error_payload: Dict[str, Any],
    scene: Dict[str, Any],
    object_names: List[str],
    failed_line_number: Optional[int] = None,
) -> str:

    obj_list_str = ", ".join(f"'{n}'" for n in object_names) if object_names else "(none)"

    if (target_robot or "").lower() == "ur5":
        skill_list = "ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper, close_ur5_ee, open_ur5_ee"
        approach_description = """# - approach must be a DICT with 'axis' (STRING in {{'x','y','z','-x','-y','-z'}}) and 'distance' (FLOAT) (do NOT use AXIS.* or enums)."""
        skill_examples = """
# Primitive call shapes (reference)
# ur5_grasp_at:  obs, reward, done = ur5_grasp_at(env, task, grasp_pos=[x,y,z], approach={{'axis':'z', 'distance':FLOAT}}, timeout_s=FLOAT)
# ur5_release_at: obs, reward, done = ur5_release_at(env, task, place_pos=[x,y,z], approach={{'axis':'z', 'distance':FLOAT}}, timeout_s=FLOAT)
# ur5_move_to:  obs, reward, done = ur5_move_to(env, task, target_pos=[x,y,z], timeout_s=FLOAT)  # rare; ur5_grasp_at/ur5_release_at usually enough
# ur5_align_gripper:
#   q = obj.get_quaternion()
#   obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction=APPROACH_DIR, timeout_s=FLOAT)
# close_ur5_ee: obs, reward, done = close_ur5_ee(env, task, gripper_close=0.0, velocity=0.2)  # close gripper (0.0=closed, 1.0=open)
# open_ur5_ee:  obs, reward, done = open_ur5_ee(env, task, gripper_open=1.0, velocity=0.2)   # open gripper (0.0=closed, 1.0=open)

# Composition patterns (reference)
# Example 1. ur5_grasp_at then ur5_release_at
# obj = Shape('object_name')
# pos = obj.get_position()
# target_pos = pos + [0.0, 0.0, 0.08]
# obs, reward, done = ur5_grasp_at(env, task, grasp_pos=pos, approach={{'axis':____, 'distance':____}}, timeout_s=____)
# obs, reward, done = ur5_release_at(env, task, place_pos=target_pos, approach={{'axis':____, 'distance':____}}, timeout_s=____)
#
# Example 2. ur5_move_to() then, ur5_align_gripper then, ur5_grasp_at/ur5_release_at
# above_grasp_pos = pos + [0.0, 0.0, 0.08]
# obs, reward, done = ur5_move_to(env, task, target_pos=above_grasp_pos, timeout_s=FLOAT)
# q = obj.get_quaternion()
# obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode=____, approach_direction=____, timeout_s=____)
"""

    elif (target_robot or "").lower() == "jaco":
        skill_list = "jaco_move_linear, jaco_move_arc, jaco_rotate_gripper, jaco_open_gripper, jaco_close_gripper, jaco_align_gripper"
        skill_examples = """
# Primitive call shapes (reference)

"""
    elif (target_robot or "").lower() == "sawyer":
        skill_list = "sawyer_move_to, sawyer_pick, sawyer_place, sawyer_align_gripper, sawyer_open_gripper, sawyer_close_gripper"
        skill_examples = """
# - sawyer_align_gripper is NOT A DEFAULT STEP. DO NOT USE it.
# - Strictly follow Composition patterns (reference).

# Primitive call shapes (reference)
# sawyer_pick:  obs, reward, done = sawyer_pick(env, task, target_object=obj, target_pos=[x,y,z], grasp_offset=[dx,dy,dz])
# sawyer_place: obs, reward, done = sawyer_place(env, task, place_pos=[x,y,z], place_offset=[dx,dy,dz])
# sawyer_move_to: obs, reward, done = sawyer_move_to(env, task, target_pos=[x,y,z], timeout_s=FLOAT)
# sawyer_open_gripper:  obs, reward, done = sawyer_open_gripper(env, task, amount=1.0, velocity=0.2)
# sawyer_close_gripper: obs, reward, done = sawyer_close_gripper(env, task, amount=0.0, velocity=0.2)

# Utility functions available (reference):
# - get_bbox_sizes(obj): returns (sizes, bbox) where sizes=[dx,dy,dz]
# - quat_mul(q1, q2): quaternion multiplication
# - normalize_vector(v): normalize vector with error handling


# Composition patterns (reference)
# Example 1. pick then lift (do not use sawyer_align_gripper)
# target = Shape('target_name'); target_pos = target.get_position()
# success = ProximitySensor('success'); success_pos = success.get_position()
# above_target_pos = target_pos + [0.0, 0.0, 0.0]
# obs, reward, done = sawyer_pick(env, task, target_object=target, target_pos=target_pos, grasp_offset=np.array([0.0, 0.0, 0.0]))
# obs, reward, done = sawyer_move_to(env, task, target_pos=success_pos, timeout_s=10.0)
"""

    else: 
        skill_list = "move, pick, place, push, open_gripper, close_gripper, align_to_quaternion, align_two_axes"
        skill_examples = """
# Primitive call shapes (reference)
# pick:  obs, reward, done = pick(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
# place: obs, reward, done = place(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
# move:  obs, reward, done = move(env, task, target_pos=[x,y,z], timeout=FLOAT)
# push:  obs, reward, done = push(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
# grip:  obs, reward, done = open_gripper(env, task) / close_gripper(env, task)
# align:
#   q = obj.get_quaternion()
#   obs, reward, done = align_to_quaternion(env, task, quaternion=q, yaw_align='parallel', approach_dir=APPROACH_DIR, timeout=FLOAT)
"""

    control_flow_rule = "# - DO NOT use if done: / else branching or other control-flow constructs (e.g., while loops)."

    # Robot-specific rules
    if (target_robot or "").lower() == "ur5":
        robot_specific_rules = """# - ur5_grasp_at/ur5_release_at already include moving/approach internally. Do NOT add extra ur5_move_to() unless truly needed.
# - Unless there is a specific reason, use only 'z' or '-z' for approach['axis'] (avoid 'x'/'y' approaches) for ur5_grasp_at.
# - If "GRASP_GUIDANCE" is provided in the scene/error payload, implement it in code exactly
# - Before calling ur5_align_gripper, first move the gripper to a safe pre-grasp pose (e.g., above_grasp_pos) using ur5_move_to().
# - For push-related tasks, call close_ur5_ee() before any push/press motion.
# - ur5_align_gripper is NOT a default step. Use it ONLY when top-down approach is not possible.
# - yaw_mode MUST be either 'parallel' or 'perpendicular' (no other values allowed)"""
    elif (target_robot or "").lower() == "sawyer":
        robot_specific_rules = """# - sawyer_pick/sawyer_place already include approach/retract internally. Do NOT add extra sawyer_move_to() before pick/place unless truly needed.
# - sawyer_align_gripper is NOT a default step. DO NOT USE it unless specifically required.
# - For push-related tasks, call sawyer_close_gripper() before any push/press motion.
# - STRICTLY FOLLOW target_pos=target_pos, grasp_offset=np.array([0.0, 0.0, 0.0]) for sawyer_pick."""
    elif (target_robot or "").lower() == "jaco":
        robot_specific_rules = """# - jaco_move_linear is for straight-line motions; jaco_move_arc for curved paths.
# - Always call jaco_open_gripper/jaco_close_gripper explicitly before/after grasping."""
    else:
        robot_specific_rules = """# - If "GRASP_GUIDANCE" is provided in the scene/error payload, implement it in code exactly"""

    prompt = f"""# Task: {task_name}
# You MUST use the EXACT objects and skills provided. NO placeholders, NO new APIs.
#
## Current state
# REPAIR: While in this state, execution failed at runtime. Your objective is to propose a localized fix that addresses the failure with minimal changes. Use the hypothesis, REFERENCE CODE, failed statement, error signal, and observations to infer the most likely cause, then output ONLY a continuation patch that fixes the issue and allows execution to proceed from the given resume point.
#
# Available objects: {obj_list_str}
# Available primitive skills: {skill_list}

# RULES (IMPORTANT)
{control_flow_rule}
# - Every primitive skill call MUST be written as: obs, reward, done = <skill>(...)
# - Never access env.scene or env.scene.objects (it does not exist).
{robot_specific_rules}
# - Use a longer timeout_s than 5.0 when needed (e.g., 8.0-15.0) to allow sufficient time for waypoint motion and avoid premature failures.
# - Use reasonable values: approach distance ~ 0.05-0.15, timeout_s ~ 8.0-15.0.
# - offset is applied when computing target_pos: target_pos = object_pos + [dx, dy, dz]
#   Example: target_pos = pos + [0.0, 0.0, 0.2]  # 20cm above
# - Output ONLY executable Python statements. ABSOLUTELY NO comments: never output '#' anywhere (even inline).
# - NEVER output any tag-like text (no angle brackets '<' or '>'), and never output the substring 'tool_call'. Only Python statements.

{skill_examples}
"""

    return prompt



# Example 1. pick then place (do not use sawyer_align_gripper)
# obj = Shape('object_name'); pos = obj.get_position()
# success = ProximitySensor('success'); success_pos = success.get_position()
# above_place_pos = success_pos + [0.0, 0.0, 0.1]
# obs, reward, done = sawyer_pick(env, task, target_object=obj, target_pos=pos, grasp_offset=np.array([0.0, 0.0, 0.0]))
# obs, reward, done = sawyer_place(env, task, place_pos=above_place_pos)
#
# Example 2. close gripper then move(push)
# approach_pos = pos + [0.0, 0.0, 0.1]
# obs, reward, done = sawyer_close_gripper(env, task)
# obs, reward, done = sawyer_move_to(env, task, target_pos=approach_pos)
# obs, reward, done = sawyer_move_to(env, task, target_pos=pos)


# -------------------------
# Code patching helpers (append / replace run_skill body)
# -------------------------

import re

_CTRL_HDR_RE = re.compile(
    r"^\s*(if|elif|for|while|try|with|except|else|finally)\b.*:\s*(#.*)?$"
)

_DEDENT_CLAUSES = ("elif", "else:", "except", "finally:")

def _indent_width(line: str) -> int:
    # tabs는 4칸으로 취급(환경에 맞게 조정 가능)
    n = 0
    for ch in line:
        if ch == " ":
            n += 1
        elif ch == "\t":
            n += 4
        else:
            break
    return n

def _is_dedent_clause(stripped: str) -> bool:
    s = stripped.lstrip()
    return (
        s.startswith("elif ")
        or s.startswith("else:")
        or s.startswith("except")
        or s.startswith("finally:")
    )

def fix_control_flow_indentation(code: str, indent_step: int = 4) -> str:
    """
    LLM이 아래처럼 내는 케이스를 보정:
      if done:
      obs, reward, done = ...
      else:
      print(...)

    원칙:
    - ':'로 끝나는 컨트롤 플로우 헤더 다음 "첫 non-empty 줄"이
      같은 indent(<= header indent)이면, 그 줄(및 연속 줄)을 +4칸
    - 헤더 다음이 바로 else/elif/except/finally로 오면, 빈 블록이므로 pass 삽입
    """
    lines = code.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped or not stripped.endswith(":") or not _CTRL_HDR_RE.match(line):
            i += 1
            continue

        hdr_indent = _indent_width(line)

        # 다음 non-empty 줄 찾기
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break

        next_line = lines[j]
        next_stripped = next_line.strip()
        next_indent = _indent_width(next_line)

        # 헤더 다음이 바로 else/elif/except/finally이면: 빈 블록 -> pass 삽입
        if next_indent == hdr_indent and _is_dedent_clause(next_stripped):
            lines.insert(j, " " * (hdr_indent + indent_step) + "pass")
            # 삽입했으니 헤더 이후는 해결된 것으로 보고 진행
            i += 1
            continue

        # 헤더 다음 줄이 같은/더 얕은 indent이면 -> 블록으로 보정
        if next_indent <= hdr_indent:
            k = j
            while k < len(lines):
                if not lines[k].strip():
                    k += 1
                    continue

                cur_indent = _indent_width(lines[k])
                cur_stripped = lines[k].strip()

                # 블록을 벗어나는 조건: 더 얕아짐
                if cur_indent < hdr_indent:
                    break

                # 같은 indent에서 dedent clause 만나면 블록 종료
                if cur_indent == hdr_indent and _is_dedent_clause(cur_stripped):
                    break

                # 같은 indent인 "블록 내부로 들어가야 할 줄"만 한 단계 indent 추가
                if cur_indent == hdr_indent:
                    lines[k] = (" " * indent_step) + lines[k]

                # 이미 더 깊게 indent된 줄은 그대로 둠
                k += 1

        i += 1

    return "\n".join(lines).strip("\n")

# -------------------------
# LLM Engine (single model, shared)
# -------------------------

def extract_function_body(code_str, func_name):
    tree = ast.parse(code_str)
    lines = code_str.splitlines(keepends=True)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # body의 첫 줄 ~ 마지막 줄
            start = node.body[0].lineno - 1
            end = node.body[-1].end_lineno

            body = "".join(lines[start:end])
            return body

    raise ValueError(f"Function '{func_name}' not found")


class LLMEngine:
    def __init__(self, model_dir: str, quant: str = "8bit", seed: int = 42):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        self.seed = seed

        self.model_dir = model_dir
        self.quant = quant
        self._gen_lock = threading.Lock()

        qcfg = None
        if quant == "8bit":
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "4bit":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif quant == "none":
            qcfg = None
        else:
            raise ValueError(f"Unsupported quant: {quant}")

        print(f"[LLM] torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[LLM] visible_gpus={torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"[LLM]  gpu[{i}]={torch.cuda.get_device_name(i)}")

        print(f"[LLM] loading tokenizer: {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, trust_remote_code=True)

        print(f"[LLM] loading model: {model_dir} (quant={quant})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map={"":0},
            quantization_config=qcfg,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        print("[LLM] model ready.")

    def _fix_seed(self, seed: int):
        import random
        import numpy as np
        import torch
        from transformers import set_seed as hf_set_seed

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # transformers 내부도 맞춰줌
        hf_set_seed(seed)

    def generate_chat(self, prompt: str, gen: Optional[Dict[str, Any]] = None) -> Tuple[str, int, float]:
        import torch

        gen = gen or {}
        seed = int(gen.get("seed", self.seed))

        min_new_tokens = int(gen.get("min_new_tokens", 128))
        max_new_tokens = int(gen.get("max_new_tokens", 800))
        do_sample = bool(gen.get("do_sample", False))
        temperature = float(gen.get("temperature", 0.0))
        top_p = float(gen.get("top_p", 1.0))
        # repetition_penalty = float(gen.get("repetition_penalty", 1.05))

        # Base model: raw text completion (no chat template)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        with self._gen_lock:
            self._fix_seed(seed)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    min_new_tokens=min_new_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    # repetition_penalty=repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        dt = time.time() - t0

        seq = out[0]
        in_len = inputs["input_ids"].shape[-1]
        gen_tokens = seq[in_len:]
        decoded = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        # decoded = extract_code_like(decoded)
        # decoded = extract_function_body(decoded,"run_skill")
        return decoded, int(gen_tokens.shape[-1]), dt

# -------------------------
# Request handling
# -------------------------

def build_generate_prompt(req: Dict[str, Any]) -> str:
    # Provider에서 prompt + reference_code + objects + descriptions 등을 보내므로
    # 서버에서 최대한 깔끔히 조립
    prompt = str(req.get("prompt", "") or "")
    reference_code = str(req.get("reference_code", "") or "")
    # reference_code 전처리 (단순 버전)
    lines = reference_code.splitlines()

    # import 제거
    lines = [
        l for l in lines
        if not l.lstrip().startswith(("import ", "from "))
    ]

    # run_skill 함수 정의부터 끝까지 사용
    filtered_code = []
    record = False
    for l in lines:
        if l.lstrip().startswith("def run_skill"):
            record = True
        if record:
            filtered_code.append(l)

    preprocessed_reference_code = "\n".join(filtered_code).strip() + "\n"
    reference_code = preprocessed_reference_code
    descriptions = str(req.get("descriptions", "") or "")
    available_objects = req.get("available_objects", []) or []
    grasp_guidance = req.get("grasp_guidance", None)

    if not isinstance(available_objects, list):
        available_objects = []

    target_scene_info = req.get("target_scene_info", None)

    # reference_code와 grasp_guidance를 prompt 내 "# Composition patterns" 직전에 삽입
    # 순서: ... reference code → grasp guidance → Composition patterns ...
    if "# Composition patterns" in prompt:
        insert_block = ""

        # 1) reference code 삽입
        if reference_code.strip():
            ref_code_safe = _comment_out_run_skill_body(reference_code)
            insert_block += "\n# REFERENCE CODE (oracle, body-only; do not copy)\n" + ref_code_safe.strip() + "\n# (end reference)\n\n"

        # 2) grasp guidance 삽입 (sawyer는 제외)
        robot_type_for_gg = str(req.get("robot_type", "ur5")).lower()
        if grasp_guidance and grasp_guidance.strip() and robot_type_for_gg != "sawyer":
            insert_block += "            # " + grasp_guidance.strip() + "\n\n"

        if insert_block:
            prompt = prompt.replace(
                "# Composition patterns",
                insert_block + "# Composition patterns",
                1,  # 첫 번째 매치만
            )
    elif reference_code.strip():

        ref_code_safe = _comment_out_run_skill_body(reference_code)
        prompt = prompt.rstrip() + "\n\n# REFERENCE CODE (oracle, body-only; do not copy)\n" + ref_code_safe.strip() + "\n# (end reference)\n"

    parts = []
    if prompt.strip():
        parts.append(prompt.strip())

    # if descriptions.strip():
    #     parts.append("\n# DESCRIPTIONS\n" + descriptions.strip())

    # if available_objects:
    #     parts.append("\n# AVAILABLE OBJECTS\n" + ", ".join([f"'{x}'" for x in available_objects]))

    # Target scene info를 Python 주석 형태로 변환
    # NOTE: 주석처리 — position을 프롬프트에 넣으면 LLM이 좌표를 하드코딩하여
    #        list+list concatenation 버그 발생. Shape().get_position() 사용을 유도하기 위해 제거.
    # if target_scene_info and isinstance(target_scene_info, list):
    #     scene_lines = ["# Target scene objects:"]
    #     for obj_info in target_scene_info:
    #         if isinstance(obj_info, dict):
    #             name = obj_info.get("name", "unknown")
    #             pos = obj_info.get("position")
    #             if pos and isinstance(pos, (list, tuple)) and len(pos) >= 3:
    #                 scene_lines.append(f"#   {name}: position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    #             else:
    #                 scene_lines.append(f"#   {name}")
    #     if len(scene_lines) > 1:
    #         parts.append("\n" + "\n".join(scene_lines))
    # elif target_scene_info and isinstance(target_scene_info, dict):
    #     scene_lines = ["# Target scene info:"]
    #     for k, v in target_scene_info.items():
    #         scene_lines.append(f"#   {k}: {v}")
    #     parts.append("\n" + "\n".join(scene_lines))

    # Add instruction for Instruct model to prevent markdown code fence output
    parts.append("\n# IMPORTANT OUTPUT FORMAT:")
    parts.append("# - Output ONLY the Python code for run_skill function body")
    parts.append("# - DO NOT use markdown code fences (```) in your output")
    parts.append("# - DO NOT add 'END SOLUTION' or similar markers")
    parts.append("# - Start directly with the code statements")

    return "\n".join(parts).strip() + "\n"

def handle_request(engine: LLMEngine, req: Dict[str, Any], addr=None) -> Dict[str, Any]:
    rtype = req.get("type")

    if rtype == "ping":
        return {"type": "pong", "ok": True}
    if rtype == "close":
        return {"type": "close_ack", "ok": True}

    # -------------------------
    # generate
    # -------------------------
    if rtype == "generate":
        # print(f"[SOCKET_SERVER] {addr} -> {rtype}")

        try:
            gen = req.get("gen", {}) or {}
            if not isinstance(gen, dict):
                gen = {}
            # initial generation can be a bit longer
            gen.setdefault("do_sample", False)
            gen.setdefault("temperature", 0.0)
            gen.setdefault("top_p", 1.0)
            gen.setdefault("min_new_tokens", 50)
            gen.setdefault("max_new_tokens", 512)

            prompt = build_generate_prompt(req)
            if _DEBUG_RPC:
                print("\n[PROMPT_TO_LLM] rtype=", rtype)
                print("[PROMPT_FULL]\n", prompt)
                print("... (len=", len(prompt), ")")

            out_text, new_tokens, dt = engine.generate_chat(prompt, gen)

            return {
                "type": "generate_result",
                "ok": True,
                "code": out_text,
                "num_tokens": new_tokens,
                "meta": {"latency_sec": dt, "new_tokens": new_tokens, "prompt": prompt},
            }
        except Exception as e:
            return {"type": "generate_result", "ok": False, "error": str(e)}

    # -------------------------
    # revise (unfinished / step revise)
        # rlbench_loop: repair_on_not_finished
    # -------------------------
    if rtype == "revise":
        try:
            task_name = str(req.get("task_name", "unknown_task"))
            robot_type = str(req.get("robot_type", "ur5"))

            inputs = req.get("inputs") or {}
            if not isinstance(inputs, dict):
                inputs = {}

            last_step = inputs.get("last_step", {}) or {}
            if not isinstance(last_step, dict):
                last_step = {}

            current_code = str(inputs.get("current_code", "") or "")

            # gate 먼저
            need_more_steps = bool(last_step.get("need_more_steps", False))
            if not need_more_steps:
                return {"type": "revise_result", "ok": True, "code": current_code, "num_tokens": 0}

            executed_stmt_texts = inputs.get("executed_stmt_texts", []) or []
            if not isinstance(executed_stmt_texts, list):
                executed_stmt_texts = [str(executed_stmt_texts)]

            scene = inputs.get("scene", {}) or {}
            if not isinstance(scene, dict):
                scene = {}

            objs = scene.get("objects", []) or []
            if not isinstance(objs, list):
                objs = []
            object_names = [o.get("name", "") for o in objs if isinstance(o, dict) and o.get("name")]

            confirmed_state = last_step.get("confirmed_state", {}) or {}

            # --- Base model: 코드 완성 패턴으로 프롬프트 구성 ---
            # 1) Rules + skill shapes (이미 전부 # 주석 형태)
            base_prompt = get_revise_prompt(
                task_name=task_name,
                target_robot=robot_type,
                scene=scene,
                executed_stmt_texts=executed_stmt_texts,
                last_step_payload=last_step,
                object_names=object_names,
            )
            parts = [base_prompt.strip()]

            # 2) Reference code (oracle) — body-only, dedent 후 # 주석
            reference_code = str(inputs.get("reference_code", "") or "")
            if reference_code.strip():
                ref_commented = _comment_out_run_skill_body(reference_code)
                if ref_commented.strip():
                    if len(ref_commented) > 20000:
                        ref_commented = ref_commented[:20000] + "\n# ... (truncated)"

                    # MeatOffGrill 태스크일 때 reference code 주석 강화
                    if task_name.lower() in ["meatoffgrill", "meatongrill"]:
                        parts.append("\n# ========================================")
                        parts.append("# CRITICAL: You MUST follow the exact pattern in the reference code below.")
                        parts.append("# ========================================")
                        parts.append("\n# REFERENCE CODE (FOLLOW THIS PATTERN EXACTLY):\n" + ref_commented.strip() + "\n# (end reference)")
                    else:
                        parts.append("\n# REFERENCE CODE (oracle, body-only; do not copy)\n" + ref_commented.strip() + "\n# (end reference)")

            # 3) Grasp guidance — # 주석 (sawyer는 제외)
            grasp_guidance = inputs.get("grasp_guidance", None)
            if grasp_guidance and str(grasp_guidance).strip() and robot_type.lower() != "sawyer":
                parts.append(f"\n# Grasp guidance: {grasp_guidance}")

            # 4) Error context (revise: task incomplete)
            parts.append(f"\n# ERROR: All {len(executed_stmt_texts)} statements were executed, but the task did not succeed.")

            # 5) 실행된 코드
            executed_block = "\n".join(str(s) for s in executed_stmt_texts).strip()

            if executed_block:
                parts.append("\n# PREVIOUSLY EXECUTED CODE (FAILED to complete the task — do NOT repeat):")
                for el in executed_block.splitlines():
                    line = el.strip()
                    if line:
                        parts.append(f"# {line}")

            # 6) Scene description - 간결하게 (물체 좌표)
            gripper = scene.get("gripper", {})
            gripper_pos = gripper.get("position")
            is_open = gripper.get("is_open", True)
            held_obj = confirmed_state.get("held_object") if confirmed_state else None

            compact_scene = []
            if gripper_pos:
                status = "OPEN" if is_open else "CLOSED"
                compact_scene.append(f"# Gripper: [{gripper_pos[0]:.2f},{gripper_pos[1]:.2f},{gripper_pos[2]:.2f}] {status}")
                if held_obj:
                    compact_scene.append(f"#   holding: '{held_obj}'")

            objs = scene.get("objects", []) or []
            if objs and len(objs) <= 6:
                for obj in objs[:6]:
                    name = obj.get("name", "?")
                    pos = obj.get("position")
                    if pos:
                        compact_scene.append(f"#   {name}: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]")

            if compact_scene:
                parts.append("\n" + "\n".join(compact_scene))

            # 7) Exploration observation (hypothesis 직전에 배치)
            exploration_obs = inputs.get("exploration_obs", None)
            if exploration_obs and isinstance(exploration_obs, dict):
                import json
                # recommendation은 별도 힌트로 분리 (JSON 크기 절약)
                recommendation = exploration_obs.pop("recommendation", None)
                obs_json = json.dumps(exploration_obs, ensure_ascii=False, separators=(',', ':'))
                if len(obs_json) > 600:
                    obs_json = obs_json[:600] + "...}"
                parts.append(f"\n# Observation: {obs_json}")
                # recommendation이 있으면 자연어 힌트로 추가
                if recommendation and isinstance(recommendation, dict):
                    hint = recommendation.get("hint", "")
                    place_pos = recommendation.get("place_pos", [])
                    if hint:
                        parts.append(f"# IMPORTANT: {hint}")
                    if place_pos:
                        parts.append(f"# Use success sensor position {place_pos} as place target.")

            # 7.5) LLM-based hypothesis generation
            # parts_without_instruction = Instruction/함수시그니처 제외한 현재까지의 parts
            _skill_map = {
                "ur5": "ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper, close_ur5_ee, open_ur5_ee",
                "jaco": "jaco_move_linear, jaco_move_arc, jaco_rotate_gripper, jaco_open_gripper, jaco_close_gripper, jaco_align_gripper",
                "sawyer": "sawyer_move_to, sawyer_pick, sawyer_place, sawyer_align_gripper, sawyer_open_gripper, sawyer_close_gripper",
            }
            _hyp_skill_list = _skill_map.get((robot_type or "").lower(), "move, pick, place, push, open_gripper, close_gripper, align_to_quaternion, align_two_axes")
            hypothesis_prompt = _build_hypothesis_prompt(
                base_prompt=base_prompt,
                parts_without_instruction=parts[1:],  # base_prompt 제외한 나머지 (ref code, scene, executed code 등)
                branch_type="revise",
                object_names=object_names,
                skill_list=_hyp_skill_list,
            )
            hypothesis = _generate_hypothesis_with_llm(engine, hypothesis_prompt)
            _hypothesis_prompt = hypothesis_prompt  # meta에 저장용
            _hypothesis_response = hypothesis       # meta에 저장용
            if hypothesis:
                parts.append(f"\n{hypothesis}")

            # 6) Instruction + 빈 함수 시그니처 (프롬프트 마지막)
            parts.append("\n# INSTRUCTION: Generate ONLY executable Python code to complete the task.")
            parts.append("# The previously executed code FAILED. Write CORRECTED code that fixes the issue.")
            parts.append("# DO NOT output comments, DO NOT copy the scene description above.")
            parts.append("# Output format: Python statements (e.g., obs, reward, done = sawyer_pick(...))")
            parts.append("def run_skill(env, task, descriptions=None, obs=None, variation_index=1):")

            prompt = "\n".join(parts).strip() + "\n    "

            gen = req.get("gen", {}) or {}
            if not isinstance(gen, dict):
                gen = {}
            gen.setdefault("do_sample", False)
            gen.setdefault("temperature", 0.0)
            gen.setdefault("top_p", 1.0)
            gen.setdefault("min_new_tokens", 80)
            gen.setdefault("max_new_tokens", 512)  # Increased for more code generation

            if _DEBUG_RPC:
                print("\n[PROMPT_TO_LLM] rtype=", rtype)
                print("[PROMPT_FULL]\n", prompt)
                print("... (len=", len(prompt), ")")
            out_text, new_tokens, dt = engine.generate_chat(prompt, gen)

            return {
                "type": "revise_result",
                "ok": True,
                "code": out_text,  # current_code에 붙일 "추가 statement들"만
                "code_format": "run_skill_body_cont",
                "num_tokens": new_tokens,
                "meta": {
                    "latency_sec": dt, "new_tokens": new_tokens, "prompt": prompt,
                    "hypothesis_prompt": _hypothesis_prompt,
                    "hypothesis_response": _hypothesis_response,
                },
            }

        except Exception as e:
            return {"type": "revise_result", "ok": False, "error": str(e)}

    # -------------------------
    # repair (failure)
        # rlbench_loop: repair_on_failure
    # -------------------------
    if rtype == "repair":
        try:
            task_name = str(req.get("task_name", "unknown_task"))
            robot_type = str(req.get("robot_type", "ur5"))

            inputs = req.get("inputs") or {}
            if not isinstance(inputs, dict):
                inputs = {}

            meta = req.get("meta", {}) or {}
            constraints = req.get("constraints", {}) or {}

            executed_stmt_texts = inputs.get("executed_stmt_texts", []) or []
            if not isinstance(executed_stmt_texts, list):
                executed_stmt_texts = [str(executed_stmt_texts)]

            failed_stmt_text = str(inputs.get("failed_stmt_text", "") or "")

            error_payload = inputs.get("error", {}) or {}
            if not isinstance(error_payload, dict):
                error_payload = {"error": str(error_payload), "error_type": "Exception"}

            scene = inputs.get("scene", {}) or {}
            if not isinstance(scene, dict):
                scene = {}

            objs = scene.get("objects", []) or []
            if not isinstance(objs, list):
                objs = []
            object_names = [o.get("name", "") for o in objs if isinstance(o, dict) and o.get("name")]

            # --- Base model: 코드 완성 패턴으로 프롬프트 구성 ---
            # 1) Rules + skill shapes (이미 전부 # 주석 형태)
            base_prompt = get_repair_prompt(
                task_name=task_name,
                target_robot=robot_type,
                current_code="",
                executed_stmt_texts=executed_stmt_texts,
                failed_stmt_text=failed_stmt_text,
                error_payload=error_payload,
                scene=scene,
                object_names=object_names,
                failed_line_number=None,
            )
            parts = [base_prompt.strip()]

            # 2) Reference code (oracle) — body-only, dedent 후 # 주석
            reference_code = str(inputs.get("reference_code", "") or "")
            if reference_code.strip():
                ref_commented = _comment_out_run_skill_body(reference_code)
                if ref_commented.strip():
                    if len(ref_commented) > 20000:
                        ref_commented = ref_commented[:20000] + "\n# ... (truncated)"

                    # MeatOffGrill 태스크일 때 reference code 주석 강화
                    if task_name.lower() in ["meatoffgrill", "meatongrill"]:
                        parts.append("\n# ========================================")
                        parts.append("# CRITICAL: For MeatOffGrill, you MUST follow the exact pattern in the reference code below.")
                        parts.append("# ========================================")
                        parts.append("\n# REFERENCE CODE (FOLLOW THIS PATTERN EXACTLY):\n" + ref_commented.strip() + "\n# (end reference)")
                    else:
                        parts.append("\n# REFERENCE CODE (oracle, body-only; do not copy)\n" + ref_commented.strip() + "\n# (end reference)")

            # 3) Grasp guidance — # 주석 (sawyer는 제외)
            grasp_guidance = inputs.get("grasp_guidance", None)
            if grasp_guidance and str(grasp_guidance).strip() and robot_type.lower() != "sawyer":
                parts.append(f"\n# Grasp guidance: {grasp_guidance}")

            # 4) Error info - 간결하게
            error_type = error_payload.get("error_type", "Exception") if isinstance(error_payload, dict) else "Exception"
            error_msg = error_payload.get("error", str(error_payload)) if isinstance(error_payload, dict) else str(error_payload)
            if len(error_msg) > 150:
                error_msg = error_msg[:150] + "..."
            parts.append(f"\n# ERROR: {error_type}: {error_msg}")

            if failed_stmt_text.strip():
                failed_short = failed_stmt_text.strip().replace('\n', ' ')
                if len(failed_short) > 80:
                    failed_short = failed_short[:80] + "..."
                parts.append(f"# Failed statement: {failed_short}")

            # PathOutOfWorkspace 에러 시 워크스페이스 가이드 추가
            if "PathOutOfWorkspace" in error_type or "path planning" in error_msg.lower():
                parts.append("# WORKSPACE HINT: The target position is outside the robot's reachable workspace (~0.8m radius).")
                parts.append("# Use positions closer to the scene objects. Avoid large offsets (> 0.15m in x/y) from object positions.")
                # scene에서 object 위치 범위를 계산하여 힌트 제공
                objs = scene.get("objects", []) or []
                if objs:
                    positions = [o.get("position") for o in objs if o.get("position")]
                    if positions:
                        import numpy as _np
                        pos_arr = _np.array(positions)
                        x_range = [float(pos_arr[:, 0].min()), float(pos_arr[:, 0].max())]
                        y_range = [float(pos_arr[:, 1].min()), float(pos_arr[:, 1].max())]
                        z_range = [float(pos_arr[:, 2].min()), float(pos_arr[:, 2].max())]
                        parts.append(f"# Scene object range: x=[{x_range[0]:.2f},{x_range[1]:.2f}], y=[{y_range[0]:.2f},{y_range[1]:.2f}], z=[{z_range[0]:.2f},{z_range[1]:.2f}]")
                        parts.append("# Keep ALL target positions within or near this range.")

            # 5) 실행된 코드
            executed_block = "\n".join(str(s) for s in executed_stmt_texts).strip()

            if executed_block:
                parts.append("\n# PREVIOUSLY EXECUTED CODE (encountered error — do NOT repeat):")
                for el in executed_block.splitlines():
                    line = el.strip()
                    if line:
                        parts.append(f"# {line}")

            # 6) Scene description - 간결하게 (물체 좌표)
            gripper = scene.get("gripper", {})
            gripper_pos = gripper.get("position")
            is_open = gripper.get("is_open", True)

            compact_scene = []
            if gripper_pos:
                status = "OPEN" if is_open else "CLOSED"
                compact_scene.append(f"# Gripper: [{gripper_pos[0]:.2f},{gripper_pos[1]:.2f},{gripper_pos[2]:.2f}] {status}")

            objs = scene.get("objects", []) or []
            if objs and len(objs) <= 6:
                for obj in objs[:6]:
                    name = obj.get("name", "?")
                    pos = obj.get("position")
                    if pos:
                        compact_scene.append(f"#   {name}: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]")

            if compact_scene:
                parts.append("\n" + "\n".join(compact_scene))

            # 6.5) Exploration observation
            exploration_obs = inputs.get("exploration_obs", None)
            if exploration_obs and isinstance(exploration_obs, dict):
                import json
                # recommendation은 별도 힌트로 분리 (JSON 크기 절약)
                recommendation = exploration_obs.pop("recommendation", None)
                obs_json = json.dumps(exploration_obs, ensure_ascii=False, separators=(',', ':'))
                if len(obs_json) > 600:
                    obs_json = obs_json[:600] + "...}"
                parts.append(f"\n# Observation: {obs_json}")
                # recommendation이 있으면 자연어 힌트로 추가
                if recommendation and isinstance(recommendation, dict):
                    hint = recommendation.get("hint", "")
                    place_pos = recommendation.get("place_pos", [])
                    if hint:
                        parts.append(f"# IMPORTANT: {hint}")
                    if place_pos:
                        parts.append(f"# Use success sensor position {place_pos} as place target.")

            # 6.6) LLM-based hypothesis generation
            _skill_map = {
                "ur5": "ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper, close_ur5_ee, open_ur5_ee",
                "jaco": "jaco_move_linear, jaco_move_arc, jaco_rotate_gripper, jaco_open_gripper, jaco_close_gripper, jaco_align_gripper",
                "sawyer": "sawyer_move_to, sawyer_pick, sawyer_place, sawyer_align_gripper, sawyer_open_gripper, sawyer_close_gripper",
            }
            _hyp_skill_list = _skill_map.get((robot_type or "").lower(), "move, pick, place, push, open_gripper, close_gripper, align_to_quaternion, align_two_axes")
            hypothesis_prompt = _build_hypothesis_prompt(
                base_prompt=base_prompt,
                parts_without_instruction=parts[1:],  # base_prompt 제외한 나머지
                branch_type="repair",
                object_names=object_names,
                skill_list=_hyp_skill_list,
            )
            hypothesis = _generate_hypothesis_with_llm(engine, hypothesis_prompt)
            _hypothesis_prompt = hypothesis_prompt  # meta에 저장용
            _hypothesis_response = hypothesis       # meta에 저장용
            if hypothesis:
                parts.append(f"\n{hypothesis}")

            # 7) Instruction + 빈 함수 시그니처 (프롬프트 마지막)
            parts.append("\n# INSTRUCTION: Fix the error and generate ONLY executable Python code.")
            parts.append("# The previously executed code FAILED. Write CORRECTED code that fixes the issue.")
            parts.append("# DO NOT output comments, DO NOT copy the error message or scene description.")
            parts.append("# Output format: Python statements (e.g., obs, reward, done = sawyer_pick(...))")
            parts.append("def run_skill(env, task, descriptions=None, obs=None, variation_index=1):")

            prompt = "\n".join(parts).strip() + "\n    "

            gen = req.get("gen", {}) or {}
            if not isinstance(gen, dict):
                gen = {}
            gen.setdefault("do_sample", False)
            gen.setdefault("temperature", 0.0)
            gen.setdefault("top_p", 1.0)
            gen.setdefault("min_new_tokens", 80)
            gen.setdefault("max_new_tokens", 512)  # Increased for more code generation

            if _DEBUG_RPC:
                print("\n[PROMPT_TO_LLM] rtype=", rtype)
                print("[PROMPT_FULL]\n", prompt)
                print("... (len=", len(prompt), ")")
            out_text, new_tokens, dt = engine.generate_chat(prompt, gen)

            return {
                "type": "repair_result",
                "ok": True,
                "code": out_text,
                "code_format": "run_skill_body_cont",
                "num_tokens": new_tokens,
                "meta": {
                    "latency_sec": dt, "new_tokens": new_tokens, "prompt": prompt,
                    "hypothesis_prompt": _hypothesis_prompt,
                    "hypothesis_response": _hypothesis_response,
                },
            }

        except Exception as e:
            return {"type": "repair_result", "ok": False, "error": str(e)}

    return {"type": "error", "ok": False, "error": f"unknown type: {rtype}"}

# -------------------------
# Client handler
# -------------------------

_DEBUG_RPC = os.environ.get("RPC_DEBUG", "0") == "1"

def _shorten(s: str, n: int = 800):
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else (s[:n] + f"\n... <truncated {len(s)-n} chars>")

def _debug_print_req(req: dict, addr=None):
    if not _DEBUG_RPC:
        return
    safe = dict(req)
    if isinstance(safe.get("prompt"), str):
        safe["prompt"] = _shorten(safe["prompt"], 800)
    if isinstance(safe.get("reference_code"), str):
        safe["reference_code"] = _shorten(safe["reference_code"], 800)
    inputs = safe.get("inputs")
    if isinstance(inputs, dict):
        inputs2 = dict(inputs)
        for k in ("current_code", "failed_stmt_text"):
            if isinstance(inputs2.get(k), str):
                inputs2[k] = _shorten(inputs2[k], 800)
        safe["inputs"] = inputs2

    prefix = f"[RPC<-RECV] from={addr} " if addr else "[RPC<-RECV] "
    print("\n" + prefix + f"type={req.get('type')} keys={list(req.keys())}")
    print(json.dumps(safe, ensure_ascii=False, indent=2))


def handle_client(conn, addr, engine: LLMEngine):
    try:
        msg = recv_json(conn)
        if msg.get("type") != "hello":
            return
        send_json(conn, {"type": "hello_ack"})

        while True:
            req = recv_json(conn)
            _debug_print_req(req, addr=addr)
            rtype = req.get("type")
            # print(f"{addr} -> {rtype}")

            resp = handle_request(engine, req, addr=addr)
            send_json(conn, resp)

    except Exception as e:
        print(f"{addr} disconnected: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

# -------------------------
# Server loop
# -------------------------

def serve(host: str, port: int, engine: LLMEngine):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        print(f"[SERVER] listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_client, args=(conn, addr, engine), daemon=True).start()

# -------------------------
# Entry
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model-dir", type=str, default=None, help="Custom model directory path")
    parser.add_argument("--model-type", type=str, default="base", choices=["base", "instruct"],
                        help="Model type: 'base' for Qwen2.5-Coder-7B, 'instruct' for Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--quant", type=str, default="8bit", choices=["8bit", "4bit", "none"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic LLM generation (default: 42)")
    args = parser.parse_args()

    # Determine model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        if args.model_type == "base":
            model_dir = "Qwen/Qwen2.5-Coder-7B"
        elif args.model_type == "instruct":
            model_dir = "Qwen/Qwen2.5-Coder-7B-Instruct"
        else:
            model_dir = "Qwen/Qwen2.5-Coder-7B"

    print(f"[SERVER] Using model: {model_dir} (type: {args.model_type if not args.model_dir else 'custom'})")
    print(f"[SERVER] Deterministic mode: seed={args.seed}, temperature=0.0, do_sample=False")
    engine = LLMEngine(model_dir=model_dir, quant=args.quant, seed=args.seed)
    serve(args.host, args.port, engine)

if __name__ == "__main__":
    main()