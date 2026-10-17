import socket
import struct
import json
import threading
import argparse
import os
import inspect
import re
import ast
import textwrap
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Literal, Any


def _debug_print_prompt(full_prompt: str, tag: str, head_n: int = 100, tail_n: int = 50):
    lines = full_prompt.splitlines()
    print(f"\n[DEBUG][PROMPT][{tag}] chars={len(full_prompt)}, lines={len(lines)}")
    print(f"\n[DEBUG][PROMPT][{tag}] head:\n" + "\n".join(lines))



# =========================
# JSON framing
# =========================

def recvall(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("client disconnected")
        data += chunk
    return data


def recv_json(sock):
    header = recvall(sock, 4)
    (length,) = struct.unpack(">I", header)
    payload = recvall(sock, length)
    return json.loads(payload.decode("utf-8"))


def send_json(sock, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(data)) + data)


def _normalize_robot_type(rt: str) -> str:
    rt = (rt or "").strip().lower()
    # 서버 내부에서 filter_invalid_params가 기대하는 키로 맞추세요.
    # (예: 'robotiq'로 관리한다면 robotiq85 -> robotiq로)
    if rt in ["robotiq85", "robotiq_85", "robotiq-85"]:
        return "robotiq"
    if rt in ["gripper", "panda"]:
        return "panda"
    return rt or "panda"


# =========================
# Genesis Skill Definitions
# =========================

GENESIS_SKILLS = {
    "panda": {
        "skills": [
            "move_gripper_to", "move_to_position", "move_parallel",
            "open_gripper", "close_gripper", "pick", "place",
            "rotate_gripper", "grasp_handle", "release_handle"
        ],
        "signatures": {
            "move_gripper_to": ["env", "obj_name", "pointing_to", "depth"],
            "move_to_position": ["env", "pos", "pointing_to", "lift_clearance", "angle"],
            "move_parallel": ["env", "move_dir", "offset", "pointing_to"],
            "open_gripper": ["env"],
            "close_gripper": ["env"],
            "pick": ["env", "obj_name", "pointing_to"],
            "place": ["env", "obj_name", "pointing_to"],
            "rotate_gripper": ["env", "angle", "steps"],
            "grasp_handle": ["env", "handle_name"],
            "release_handle": ["env"],
        }
    },
    "robotiq": {
        "skills": [
            "move_gripper_to", "move_to_position", "move_parallel",
            "open_robotiq85", "close_robotiq85", "pick_robotiq85", "place_robotiq85",
            "grasp_handle_robotiq85", "release_handle_robotiq85"
        ],
        "signatures": {
            "move_gripper_to": ["env", "obj_name", "pointing_to", "depth"],
            "move_to_position": ["env", "pos", "pointing_to", "lift_clearance", "angle"],
            "move_parallel": ["env", "move_dir", "offset", "pointing_to"],
            "open_robotiq85": ["env"],
            "close_robotiq85": ["env"],
            "pick_robotiq85": ["env", "obj_name", "pointing_to"],
            "place_robotiq85": ["env", "obj_name", "pointing_to"],
            "grasp_handle_robotiq85": ["env", "handle_name"],
            "release_handle_robotiq85": ["env"],
        }
    },
    "suction": {
        "skills": [
            "move_gripper_to", "move_to_position", "move_parallel",
            "activate_vacuum", "deactivate_vacuum",
            "attach_vacuum_handle", "detach_vacuum_handle"
        ],
        "signatures": {
            "move_gripper_to": ["env", "obj_name", "pointing_to", "depth"],
            "move_to_position": ["env", "pos", "pointing_to", "lift_clearance", "angle"],
            "move_parallel": ["env", "move_dir", "offset", "pointing_to"],
            "activate_vacuum": ["env"],
            "deactivate_vacuum": ["env"],
            "attach_vacuum_handle": ["env", "handle_name"],
            "detach_vacuum_handle": ["env"],
        }
    }
}

# Subtask descriptions for repair/revise prompts
SUBTASK_DESCRIPTIONS = {
    "open_box_lid": "Open the box lid FULLY by grasping the handle and lifting it completely. The lid must be opened wide enough to access objects inside.",
    "open_hinge_lid": "Open the hinged lid FULLY by grasping the handle and lifting it completely. The lid must be opened wide enough to access objects inside.",
    "open_drawer": "Open the drawer FULLY by grasping and pulling the handle. The drawer must be pulled out completely to access objects inside.",
    "pick_single": "Pick up ONLY obj_name (use the parameter, not hardcoded names). Do NOT pick other objects.",
    "place_single": "Place a single object at the target location",
    "pick_and_place_object": "Pick up an object and place it at the target location",
    "unstack_single": "Unstack a single object from on top of another object",
    "stack_single": "Stack a single object on top of another object",
    "close_drawer": "Closes a prismatic drawer by grasping and pushing the handle",

}

# Available primitives by robot type
ROBOT_PRIMITIVES = {
    "panda": "move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, grasp_handle, release_handle",
    "robotiq": "move_gripper_to, move_to_position, move_parallel, open_robotiq85, close_robotiq85, grasp_handle_robotiq85, release_handle_robotiq85",
    "suction": "move_gripper_to, move_to_position, move_parallel, activate_vacuum, deactivate_vacuum, attach_vacuum_handle, detach_vacuum_handle, rotate_gripper",
}


def get_skills_hint(robot_type: str) -> str:
    """Get available skills string for a robot type."""
    skills = GENESIS_SKILLS.get(robot_type, GENESIS_SKILLS["panda"])["skills"]
    return ", ".join(skills)


# def _get_skills_hint(target_robot: str) -> str:
    """
    Get available skills hint based on robot type.

    Args:
        target_robot: Robot type ("panda", "robotiq", or "suction")

    Returns:
        Skills hint string
    """
    # if target_robot == "panda":
    #     return "move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, pick, place, rotate_gripper, grasp_handle, release_handle"
    # elif target_robot == "robotiq":
    #     return "move_gripper_to, move_to_position, move_parallel, open_robotiq85, close_robotiq85, pick_robotiq85, place_robotiq85, grasp_handle_robotiq85, release_handle_robotiq85"
    # else:  # suction
    #     return "move_gripper_to, move_to_position, move_parallel, activate_vacuum, deactivate_vacuum, attach_vacuum_handle, detach_vacuum_handle"



# =========================
# Scene
# =========================

def build_scene_description(
    scene: Dict[str, Any],
    gripper_state: Optional[Dict[str, Any]] = None,
    executed_skill_texts: Optional[List[str]] = None,
) -> str:
    """
    scene 정보를 자연어 형태의 묘사로 변환합니다.

    Args:
        scene: get_scene_info()에서 반환된 scene 정보
            - objects: list of {name, type, position, quaternion}
            - gripper: {position, quaternion, is_open}
        gripper_state: 추가적인 gripper 상태 정보 (held_object 등)
        executed_skill_texts: 이미 실행된 statement 리스트

    Returns:
        scene 묘사 문자열
    """
    desc_parts = []

    # Gripper 상태 묘사
    gripper = scene.get("gripper", {})
    gripper_pos = gripper.get("position")
    is_open = gripper.get("is_open", True)

    gripper_desc = "# Current gripper state:\n"
    if isinstance(gripper_pos, (list, tuple)) and len(gripper_pos) >= 3:
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
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                obj_desc += f"#   {name} ({obj_type}): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            else:
                obj_desc += f"#   {name} ({obj_type}): position unknown\n"
        desc_parts.append(obj_desc)

    # 실행된 statements 요약
    if executed_skill_texts:
        exec_desc = f"# Already executed {len(executed_skill_texts)} statement(s):\n"
        for i, stmt in enumerate(executed_skill_texts[-5:], start=max(0, len(executed_skill_texts)-5)+1):
            # 마지막 5개만 표시
            stmt_short = stmt.strip().replace('\n', ' ')[:80]
            exec_desc += f"#   [{i}] {stmt_short}{'...' if len(stmt.strip()) > 80 else ''}\n"
        if len(executed_skill_texts) > 5:
            exec_desc = f"# ... ({len(executed_skill_texts) - 5} earlier statements omitted)\n" + exec_desc.split('\n', 1)[1]
        desc_parts.append(exec_desc)

    return "\n".join(desc_parts)



# =========================
# Prompt Builders (comment-style, concise)
# =========================


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


def extract_docstring_from_ref(processed_ref: str) -> str:
    """
    processed_ref에서 첫 번째 docstring (\"\"\"...\"\"\" 또는 '''...''')을 추출.
    """
    if not processed_ref:
        return ""

    # Find triple quotes (either """ or ''')
    for quote in ['"""', "'''"]:
        start = processed_ref.find(quote)
        if start != -1:
            end = processed_ref.find(quote, start + 3)
            if end != -1:
                docstring = processed_ref[start + 3:end].strip()
                return docstring
    return ""



# 최종 INITIAL prompt
def build_generate_prompt(req: dict) -> str:
    """Build initial code generation prompt for Genesis."""
    prompt = req.get("prompt", "")
    reference_code = req.get("reference_code", "")
    scene_info = req.get("target_scene_info", [])
    robot_type = req.get("target_robot", req.get("robot_type", "panda"))

    # Handle case where scene_info is a list (take first element if it's a dict)
    if isinstance(scene_info, list):
        # 리스트는 "objects 리스트"로 간주
        scene_info = {"objects": scene_info}
    elif not isinstance(scene_info, dict):
        scene_info = {}

    # Build reference code section
    processed_ref = ""
    if reference_code:
        processed_ref = preprocess_reference_code(reference_code)

    # Build scene context
    scene_context = ""
    if scene_info:
        scene_lines = []

        # Gripper state
        gripper = scene_info.get("gripper", {})
        if gripper:
            gripper_open = gripper.get("is_open", True)
            gripper_pos = gripper.get("position", [])
            pos_str = [round(p, 3) for p in gripper_pos] if gripper_pos else "N/A"
            scene_lines.append(f"# gripper: is_open={gripper_open}, pos={pos_str}")

        # Object positions - REMOVED to prevent LLM from hardcoding object names
        # The LLM should use env.get_obj_pos(obj_name) instead
        # objects = scene_info.get("objects", [])
        # if objects:
        #     scene_lines.append("# objects:")
        #     for obj in objects[:10]:
        #         name = obj.get("name", "")
        #         pos = obj.get("position", [])
        #         if name and pos:
        #             scene_lines.append(f"#   {name}: pos={[round(p, 3) for p in pos]}")
        """
        # Graspable objects with recommended approach
        graspable = scene_info.get("graspable_objects", [])
        if graspable:
            scene_lines.append("# graspable_objects:")
            for g in graspable:
                obj_name = g.get("object_name", "")
                can_grasp = g.get("can_grasp", False)
                rec_approach = g.get("recommended_approach", {})
                if obj_name:
                    approach_str = ""
                    if rec_approach:
                        local_axis = rec_approach.get("local_axis", "")
                        world_dir = rec_approach.get("world_direction", "")
                        if local_axis and world_dir:
                            approach_str = f", recommended_approach={{axis:'{local_axis}', dir:'{world_dir}'}}"
                    scene_lines.append(f"#   {obj_name}: can_grasp={can_grasp}{approach_str}")
        """
        # [CURRENT SCENE] 뒷부분에 REFERENCE CODE도 넣어주기
        if processed_ref and processed_ref.strip():
            scene_lines.append(f"\n# [REFERENCE CODE (TASK LOGIC SPEC)]")
            scene_lines.append(f"# CRITICAL: You MUST follow this reference code's logic and helper functions EXACTLY.")
            scene_lines.append(f"# - Copy the EXACT sequence of skills (move_gripper_to, move_parallel, use_smart_placement, find_place_position etc.)")
            scene_lines.append(f"# - Preserve loop counts (e.g., 'for _ in range(3)' means repeat 3 times)")
            scene_lines.append(f"# - Keep the same parameter and values (distances, directions, pointing_to)")
            scene_lines.append(f"{processed_ref}")
            scene_lines.append(f"# [REFERENCE CODE] end\n")
        if scene_lines:
            scene_context = "\n# [CURRENT SCENE]\n" + "\n".join(scene_lines) + "\n"

    # Insert scene context before "# HARD CONSTRAINTS"
    def_pos = prompt.find("# HARD CONSTRAINTS")
    if def_pos != -1 and scene_context:
        prompt = prompt[:def_pos] + scene_context + "\n" + prompt[def_pos:]
    full_prompt = prompt

    # Insert reference docstring into def run_task's docstring (after first """ in def run_task)
    if processed_ref:
        ref_docstring = extract_docstring_from_ref(processed_ref)
        if ref_docstring:
            # Find def run_task first, then find """ after it
            run_task_pos = full_prompt.find("def run_task")
            if run_task_pos != -1:
                # Find the first """ after def run_task
                first_quote_pos = full_prompt.find('"""', run_task_pos)
                if first_quote_pos != -1:
                    # Insert the docstring right after the first """
                    insert_pos = first_quote_pos + 3
                    # Insert without extra indentation
                    full_prompt = full_prompt[:insert_pos] + "\n" + ref_docstring + "\n" + full_prompt[insert_pos:]

    if req.get("debug_prompt", False):
        # print("\n[DEBUG][PROMPT][GENERATE] full_prompt head:\n" + "\n".join(full_prompt.splitlines()[:80]))
        # print("\n[DEBUG][PROMPT][GENERATE] full_prompt tail:\n" + "\n".join(full_prompt.splitlines()[-40:]))
        print(f"\n[DEBUG][PROMPT][GENERATE] chars={len(full_prompt)}, lines={len(full_prompt.splitlines())}")
        # 특정 섹션이 들어갔는지 빠르게 체크
        for k in ["[REFERENCE CODE", "[CURRENT SCENE]", "[GRASP GUIDANCE]", "# RULES", "def run_task"]:
            print(f"[DEBUG][PROMPT][GENERATE] contains({k})={k in full_prompt}")
        print("")

    return full_prompt



def get_base_prompt(
    mode: Literal["repair", "revise"],
    task_name: str,
    target_robot: str,
    obj_list_str: str,
    scene: Dict[str, Any],
    subtask_name: str = "",
) -> str:

    if not obj_list_str or obj_list_str == "(none)":
        derived = ", ".join(
            f"'{o.get('name')}'"
            for o in (scene.get("objects", []) if isinstance(scene, dict) else [])
            if isinstance(o, dict) and o.get("name")
        )
        if derived:
            obj_list_str = derived
            
    if (target_robot or "").lower() == "robotiq":
        skill_examples = """
# Available primitives: move_gripper_to, move_to_position, move_parallel, open_robotiq85, close_robotiq85, grasp_handle_robotiq85, release_handle_robotiq85

# RULES (MUST FOLLOW)
# - Generate subtask function body only (no loops over multiple objects)
# - Use ONLY the Robotiq85 primitives listed above
# - pointing_to: 'down', 'left', or 'right' (string)
# - pick = move_gripper_to + close_robotiq85
# - place = move_to_position + open_robotiq85
# - grasp_handle_robotiq85/release_handle_robotiq85 for handles on boxes/drawers

# ROBOTIQ PRIMITIVES (MUST FOLLOW)
# move_gripper_to(env, obj_name, pointing_to='down', depth=0.01)
# move_to_position(env, pos, pointing_to='down', lift_clearance=0.12)
# move_parallel(env, direction, offset, pointing_to='down')  # direction: 'up','left','right','front','back'
# open_robotiq85(env)
# close_robotiq85(env)
# grasp_handle_robotiq85(env, handle_name)
# release_handle_robotiq85(env)

# UTILITIES
# env.get_obj_pos(name) -> [x,y,z]
# env.get_obj_bbox(name) -> [[min],[max]]
# env.is_obj_visible(name) -> bool

# HARD CONSTRAINTS (DO NOT VIOLATE)
# - IMPORTANT: Preserve the exact indentation (if/for block depth) from the reference code.
# - Write direct primitive calls sequentially (imperative code).
# - NEVER use dict/list/set comprehensions.
# - IMPORTANT: Use ACTUAL object names from the "Available objects" list.
# - NEVER use placeholder comments like '# ...' or '# TODO' or '# Your code here'
# - Output ONLY complete executable Python statements
# - Use obj_pos = env.get_obj_pos(obj_name) for needed objects.
# - IMPORTANT: You MUST only use ROBOTIQ PRIMITIVES strictly.
# - NEVER USE open_gripper, close_gripper, grasp_handle, release_handle.


# EXAMPLE 1: Pick and place
# target_pos = env.get_obj_pos(target_name)
# move_gripper_to(env, obj_name, pointing_to="down")
# close_robotiq85(env)
# move_parallel(env, "up", 0.15, pointing_to="down")
# target_pos[2] += 0.10
# move_to_position(env, target_pos, pointing_to="down")
# open_robotiq85(env)
# move_parallel(env, "up", 0.08, pointing_to="down")

# EXAMPLE 2: Open hinged lid (top-down approach)
# move_gripper_to(env, handle_name, pointing_to="down")
# grasp_handle_robotiq85(env, handle_name)
# for _ in range(5):
#     move_parallel(env, "up", 0.08, pointing_to="down")
#     move_parallel(env, "front", 0.08, pointing_to="down")
# release_handle_robotiq85(env)

# EXAMPLE 3: Open drawer (side approach)
# move_gripper_to(env, handle_name, pointing_to="right")
# grasp_handle_robotiq85(env, handle_name)
# move_parallel(env, "left", 0.12, pointing_to="right")
# release_handle_robotiq85(env)
"""

    elif (target_robot or "").lower() == "suction":
        skill_examples = """
# Available primitives: move_gripper_to, move_to_position, move_parallel, activate_vacuum, deactivate_vacuum, attach_vacuum_handle, detach_vacuum_handle, rotate_gripper

# RULES (MUST FOLLOW)
# - Generate subtask function body only (no loops over multiple objects)
# - Use ONLY the SUCTION PRIMITIVE listed above - DO NOT use any other functions!
# - pointing_to: 'down', 'left', or 'right' (string)
# - pick = move_gripper_to + activate_vacuum
# - place = move_to_position + deactivate_vacuum
# - attach_vacuum_handle/detach_vacuum_handle for handles
# - FORBIDDEN: pick_and_place_object, pick_and_place, pick_object, place_object (these do NOT exist!)
# - Use ONLY the SUCTION PRIMITIVE listed - DO NOT use any other functions!
# - DO NOT use open_gripper, close_gripper

# SUCTION PRIMITIVES (MUST FOLLOW)
# move_gripper_to(env, obj_name, pointing_to='down', depth=0.01)
# move_to_position(env, pos, pointing_to='down', lift_clearance=0.12)
# move_parallel(env, direction, offset, pointing_to='down')  # direction: 'up','left','right','front','back'
# activate_vacuum(env)
# deactivate_vacuum(env)
# attach_vacuum_handle(env, handle_name)
# detach_vacuum_handle(env)
# rotate_gripper(env, angle, steps=60)  # for pouring

# UTILITIES
# env.get_obj_pos(name) -> [x,y,z]
# env.get_obj_bbox(name) -> [[min],[max]]
# env.is_obj_visible(name) -> bool


# HARD CONSTRAINTS (DO NOT VIOLATE)
# - IMPORTANT: Preserve the exact indentation (if/for block depth) from the reference code.
# - Write direct primitive calls sequentially (imperative code).
# - NEVER use dict/list/set comprehensions.
# - IMPORTANT: Use ACTUAL object names from the "Available objects" list.
# - NEVER use placeholder comments like '# ...' or '# TODO' or '# Your code here'
# - Output ONLY complete executable Python statements
# - Use obj_pos = env.get_obj_pos(obj_name) for needed objects.
# - IMPORTANT: You MUST only use SUCTION PRIMITIVES strictly.
# - NEVER USE open_gripper, close_gripper, grasp_handle, release_handle.


# EXAMPLE 1: Pick and place (NEVER USE open_gripper, close_gripper)
# target_pos = env.get_obj_pos(target_name)
# move_gripper_to(env, obj_name, pointing_to="down")
# activate_vacuum(env)
# move_parallel(env, "up", 0.15, pointing_to="down")
# target_pos[2] += 0.10
# move_to_position(env, target_pos, pointing_to="down")
# deactivate_vacuum(env)
# move_parallel(env, "up", 0.08, pointing_to="down")

# EXAMPLE 2: Open hinged lid (top-down approach)
# move_gripper_to(env, box_handle, pointing_to="down")
# activate_vacuum(env)
# attach_vacuum_handle(env, box_handle)
# for _ in range(5):
#     move_parallel(env, "up", 0.08, pointing_to="down")
#     move_parallel(env, "front", 0.08, pointing_to="down")
# detach_vacuum_handle(env)
# deactivate_vacuum(env)

# EXAMPLE 3: Open drawer (side approach)
# move_gripper_to(env, handle_name, pointing_to="right")
# activate_vacuum(env)
# attach_vacuum_handle(env, handle_name)
# move_parallel(env, "left", 0.12, pointing_to="right")
# detach_vacuum_handle(env)
# deactivate_vacuum(env)
"""

    else: 
        skill_examples = """
# Available primitives: move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, grasp_handle, release_handle

# RULES (IMPORTANT)
# - Generate subtask function body only (no loops over multiple objects)
# - Use ONLY the primitives listed above
# - pointing_to: 'down', 'left', or 'right' (string)
# - pick = move_gripper_to + close_gripper
# - place = move_to_position + open_gripper
# - grasp_handle/release_handle for handles on boxes/drawers

# PRIMITIVES
# move_gripper_to(env, obj_name, pointing_to='down', depth=0.01)
# move_to_position(env, pos, pointing_to='down', lift_clearance=0.12)
# move_parallel(env, direction, offset, pointing_to='down')  # direction: 'up','left','right','front','back'
# open_gripper(env)
# close_gripper(env)
# grasp_handle(env, handle_name) -> close_gripper after contact
# release_handle(env) -> open_gripper and release

# UTILITIES
# env.get_obj_pos(name) -> [x,y,z]
# env.get_obj_bbox(name) -> [[min],[max]]
# env.is_obj_visible(name) -> bool

# EXAMPLE 1: Pick and place
# target_pos = env.get_obj_pos(target_name)
# move_gripper_to(env, obj_name, pointing_to="down")
# close_gripper(env)
# move_parallel(env, "up", 0.15, pointing_to="down")
# target_pos[2] += 0.10
# move_to_position(env, target_pos, pointing_to="down")
# open_gripper(env)
# move_parallel(env, "up", 0.08, pointing_to="down")

# EXAMPLE 2: Open hinged lid (top-down approach)
# move_gripper_to(env, handle_name, pointing_to="down")
# grasp_handle(env, handle_name)
# for _ in range(5):
#     move_parallel(env, "up", 0.08, pointing_to="down")
#     move_parallel(env, "front", 0.08, pointing_to="down")
# release_handle(env)

# EXAMPLE 3: Open drawer (side approach)
# move_gripper_to(env, handle_name, pointing_to="right")
# grasp_handle(env, handle_name)
# move_parallel(env, "left", 0.12, pointing_to="right")
# release_handle(env)
"""

    # subtask_desc = SUBTASK_DESCRIPTIONS.get(subtask_name, "")
    subtask_desc = ""  # Temporarily disabled

    repair_baseprompt = f"""# EXECUTION FAILURE - REPAIR NEEDED
# MANDATORY: Use these EXACT objects and skills. NO placeholders!"""

    # if subtask_desc:
    #     repair_baseprompt += f"\n# Subtask: {subtask_name} - {subtask_desc}"
    if subtask_name:
        repair_baseprompt += f"\n# Subtask: {subtask_name}"

    # Add special instruction for opening subtasks
    opening_instruction = ""
    if subtask_name in ["open_box_lid", "open_hinge_lid", "open_drawer"]:
        opening_instruction = """
# Do NOT stop after just a few movements. Continue lifting/pulling until the object is completely open."""

    repair_baseprompt += f"""
# A failure occurred while executing the policy. You must repair from the failed statement onward.
{opening_instruction}
# Available objects: {obj_list_str}
{skill_examples}

# HARD CONSTRAINTS (DO NOT VIOLATE)
# - DO NOT build action dictionaries or planners.
# - Write direct primitive calls sequentially (imperative code).
# - NEVER use dict/list/set comprehensions.
# - DO NOT iterate over env.get_obj_names() to compute positions.
# - IMPORTANT: ONLY USE the Args given. NEVER use other objects or parameters.
# - DO NOT use placeholder comments like '# ...' or '# TODO' or '# Your code here'
# - Output ONLY complete executable Python statements
#
# IMPORTANT: Only use visible objects. Do NOT call env.get_obj_pos() on objects with visible=False (e.g., liquid).
# Available primitives: {ROBOT_PRIMITIVES[target_robot]}
"""

    revise_baseprompt = f"""# TASK INCOMPLETE - NEED MORE STEPS
# MANDATORY: Use these EXACT objects and skills. NO placeholders!"""

    if subtask_desc:
        revise_baseprompt += f"\n# Subtask: {subtask_name} - {subtask_desc}"
    elif subtask_name:
        revise_baseprompt += f"\n# Subtask: {subtask_name}"

    revise_baseprompt += f"""
# The task is NOT yet complete. Generate additional code to finish the task.
{opening_instruction}
{skill_examples}

# HARD CONSTRAINTS (DO NOT VIOLATE)
# - DO NOT build action dictionaries or planners.
# - Write direct primitive calls sequentially (imperative code).
# - NEVER use dict/list/set comprehensions.
# - DO NOT iterate over env.get_obj_names() to compute positions.
# - IMPORTANT: ONLY USE the Args given. NEVER use other objects or parameters.
# - NEVER reference variables that are not function arguments or locally defined.
# - DO NOT use placeholder comments like '# ...' or '# TODO' or '# Your code here'
# - Output ONLY complete executable Python statements
#
# IMPORTANT: Only use visible objects. Do NOT call env.get_obj_pos() on objects with visible=False (e.g., liquid).
# Available primitives: {ROBOT_PRIMITIVES[target_robot]}
"""
    if mode == "repair":
        return repair_baseprompt
    else:  # mode == "revise"
        return revise_baseprompt


# - DO NOT create obj_positions / positions dicts like: obj_positions = {{obj: env.get_obj_pos(obj) for obj in env.get_obj_names()}}



# 최종 REPAIR prompt
def build_repair_prompt(req: dict) -> str:
    """Build repair prompt for failed execution in Genesis."""

    inputs = req.get("inputs", {}) or {}
    if not isinstance(inputs, dict):
        inputs = {}


    # ---- basics ----
    task_name = str(req.get("task_name", "unknown_task"))
    skill_name = str(req.get("skill_name", ""))
    robot_type = str(req.get("target_robot", req.get("robot_type", "panda")))
    scene = inputs.get("scene", {}) or {}
    if not isinstance(scene, dict):
        scene = {}

    executed_skills = inputs.get("executed_skill_texts", []) or []
    if not isinstance(executed_skills, list):
        executed_skills = [str(executed_skills)]

    failed_stmt = str(inputs.get("failed_skill_text", "") or inputs.get("failed_stmt_text") or "")
    if not failed_stmt.strip() and executed_skills:
        failed_stmt = str(executed_skills[-1])

    available_objects = inputs.get("available_objects", req.get("available_objects", [])) or []
    if not isinstance(available_objects, list):
        available_objects = [str(available_objects)]
    obj_list_str = ", ".join(f"'{n}'" for n in available_objects) if available_objects else "(none)"

    constraints = req.get("constraints", {}) or {}
    if not isinstance(constraints, dict):
        constraints = {"raw": str(constraints)}
    constraints = dict(constraints) if isinstance(constraints, dict) else {"raw": str(constraints)}
    constraints["executed_prefix_count"] = len(executed_skills)


    # ---- error payload ----
    error = inputs.get("error", {}) or {}
    if not isinstance(error, dict):
        error = {"message": str(error)}

    error_type = str(error.get("error_type", "") or error.get("skill_name", "Unknown"))
    error_msg = str(error.get("error", "") or error.get("error_message", "") or error.get("message", ""))


    # 1) BASE
    base_prompt = get_base_prompt(mode="repair", task_name=task_name, target_robot=robot_type, obj_list_str=obj_list_str, scene=scene, subtask_name=skill_name)
    parts = [base_prompt.strip()]

    # 2) SKILL LIST
    skills_hint = get_skills_hint(robot_type)
    # parts.append(skills_hint)


    # 3) REFERENCE CODE (oracle)
    reference_code = str(inputs.get("reference_code", "") or "")
    processed_ref = ""
    if reference_code:
        processed_ref = preprocess_reference_code(reference_code).strip()

    if processed_ref:
        parts.append("\n# [REFERENCE CODE (TASK LOGIC SPEC)]")
        parts.append("# CRITICAL: Follow this reference code's logic and structure EXACTLY.")
        parts.append("# - Copy the EXACT sequence of primitive calls")
        parts.append("# - Preserve loop counts and parameter values")
        parts.append(processed_ref)
        parts.append("# [REFERENCE CODE] end\n")

    # 4) SCENE DESC, EXECUTED STMTS
    scene_desc = build_scene_description(
        scene=scene,
        gripper_state=inputs.get("gripper_state"),
        executed_skill_texts=executed_skills,
    )
    if scene_desc.strip():
        if len(scene_desc) > 9000:
            scene_desc = scene_desc[:9000] + "\n# ... (truncated)"
        parts.append("\n# SCENE DESCRIPTION (IMPORTANT)\n" + scene_desc.strip() + "\n")

    # parts.append("[EXECUTED_STATEMENTS_BEGIN]")
    # parts.append(f"COUNT={len(executed_skills)}")
    # if executed_skills:
        # 빈 줄 없이, stmt들을 그대로 이어붙임 (stmt 내부의 줄바꿈은 유지됨)
    #     parts.append("\n".join(str(s).rstrip("\n") for s in executed_skills))
    # parts.append("[EXECUTED_STATEMENTS_END]")
    

    # 5) FAILED STMT, ERROR_MSG, CONSTRAINTS
    """
    if failed_stmt.strip():
        fs = failed_stmt.strip()
        if len(fs) > 6000:
            fs = fs[:6000] + "\n# ... (truncated)"
        parts.append("[FAILED_STATEMENT_BEGIN]")
        parts.append(fs)
        parts.append("[FAILED_STATEMENT_END]")
    """
    parts.append("\n# FAILED_STATEMENT:" + failed_stmt.strip() + "\n")

    err_block = []
    if error_type:
        err_block.append(f"# ERROR_TYPE: {error_type}")
    if error_msg:
        err_block.append(f"# ERROR_MSG: {error_msg}")

    tb = str(error.get("traceback", "") or "")
    if tb.strip():
        tail = "\n".join(tb.strip().splitlines()[-2:])
        err_block.append("# TRACEBACK_TAIL:\n# " + "\n# ".join(tail.splitlines()))

    if err_block:
        parts.append("\n# ERROR (IMPORTANT)\n" + "\n".join(err_block) + "\n")


    """
    try:
        c_str = json.dumps(constraints, ensure_ascii=False, indent=2)
    except Exception:
        c_str = str(constraints)
    if c_str.strip():
        if len(c_str) > 3000:
            c_str = c_str[:3000] + "\n# ... (truncated)"
        parts.append("\n# CONSTRAINTS (json)\n" + c_str.strip() + "\n")
    """


    # 6) 마지막 CONSTRAINTS 재강조 + 트리거
    parts.append(
        "\n# CONSTRAINTS:\n"
        "# The first N executed statements are already executed. DO NOT reprint the prefix. Only output new statements.\n"
        "# Write continuation statements WITHOUT extra leading indentation (start at body-level).\n"
        "# Output ONLY new Python statements (function body level).\n"
        "# DO NOT REPEAT ANY PROMPT TEXTS, headers, executed blocks, constraints json, or reference.\n"
        "# DO NOT output markdown fences.\n"
        "# From the GIVEN CURRENT STATE (scene/gripper/error context), write continuation code that COMPLETES the task.\n"
        "# DO NOT comment out code. Do NOT start lines with '#'. Output executable statements only.\n"
        "# IMPORTANT: Follow the indentation, parameters, UTILITIES and Args(find_place_position, use_smart_placement, etc.) shown in [REFERENCE CODE (TASK LOGIC SPEC)].\n"
        "\n# Now write the fixed continuation:\n"
    )

    repair_prompt = "\n".join(parts).strip() + "\n"

    return repair_prompt


        # DO NOT output anything from blocks:\n"
        #   [FAILED_STATEMENT_BEGIN]..[FAILED_STATEMENT_END]\n"
        #   [REFERENCE ...]\n"



# 최종 REVISE prompt
def build_revise_prompt(req: dict) -> str:
    """Build revise prompt for step-by-step refinement in Genesis."""

    inputs = req.get("inputs", {}) or {}
    if not isinstance(inputs, dict):
        inputs = {}

    available_objects = inputs.get("available_objects", req.get("available_objects", [])) or []
    if not isinstance(available_objects, list):
        available_objects = [str(available_objects)]
    obj_list_str = ", ".join(f"'{n}'" for n in available_objects) if available_objects else "(none)"

    
    # ---- basics ----
    task_name = str(req.get("task_name", "unknown_task"))
    skill_name = str(req.get("skill_name", ""))
    robot_type = str(req.get("target_robot", req.get("robot_type", "panda")))
    meta = str(req.get("model", req.get("source_robot")))
    scene = inputs.get("scene", {}) or {}
    if not isinstance(scene, dict):
        scene = {}

    executed_skills = inputs.get("executed_skill_texts", []) or []
    if not isinstance(executed_skills, list):
        executed_skills = [str(executed_skills)]
    last_step = inputs.get("last_step", {}) or {}
    if not isinstance(last_step, dict):
        last_step = {"raw": str(last_step)}

    """
    constraints = req.get("constraints", {}) or {}
    if not isinstance(constraints, dict):
        constraints = {"raw": str(constraints)}
    constraints = dict(constraints) if isinstance(constraints, dict) else {"raw": str(constraints)}
    # constraints["executed_prefix_count"] = len(executed_skills)
    """

    # last step payload

    # 1) BASE
    base_prompt = get_base_prompt(mode="revise", task_name=task_name, target_robot=robot_type, obj_list_str=obj_list_str, scene=scene, subtask_name=skill_name)
    parts = [base_prompt.strip()]

    # 2) SKILL LIST
    skills_hint = get_skills_hint(robot_type)
    parts.append(skills_hint)

    # 3) REFERENCE CODE (oracle)
    reference_code = str(inputs.get("reference_code", "") or "")
    processed_ref = ""
    if reference_code:
        processed_ref = preprocess_reference_code(reference_code).strip()

    if processed_ref:
        parts.append("\n# [REFERENCE CODE (TASK LOGIC SPEC)]")
        parts.append("# CRITICAL: Follow this reference code's logic and structure EXACTLY.")
        parts.append("# - Copy the EXACT sequence of primitive calls")
        parts.append("# - Preserve loop counts and parameter values")
        parts.append(processed_ref)
        parts.append("# [REFERENCE CODE] end\n")
    parts.append("# OUTPUT ONLY VALID PYTHON CODE. NO JSON, NO MARKDOWN, NO EXPLANATIONS.\n")


    # 4) SCENE DESC, EXECUTED STMTS
    scene_desc = build_scene_description(
        scene=scene,
        gripper_state=inputs.get("gripper_state"),
        executed_skill_texts=executed_skills,
    )
    if scene_desc.strip():
        if len(scene_desc) > 9000:
            scene_desc = scene_desc[:9000] + "\n# ... (truncated)"
        parts.append("\n# SCENE DESCRIPTION\n" + scene_desc.strip() + "\n")

    # Executed statements section (without confusing markers)
    if executed_skills:
        parts.append("\n# [ALREADY EXECUTED - DO NOT REPEAT]")
        parts.append("\n".join(str(s).rstrip("\n") for s in executed_skills))
        parts.append("# [END OF EXECUTED CODE]\n")
    

    # 5) LAST STEP, CONSTRAINTS
    # LAST_STEP JSON을 주석으로 표시하여 LLM이 코드로 착각하지 않도록 함
    try:
        last_step_str = json.dumps(last_step, ensure_ascii=False, indent=2)
    except Exception:
        last_step_str = str(last_step)
    if last_step_str.strip():
        if len(last_step_str) > 6000:
            last_step_str = last_step_str[:6000] + "\n# ... (truncated)"
        # 각 줄 앞에 # 추가하여 주석 처리
        commented_last_step = "\n".join("# " + line for line in last_step_str.strip().splitlines())
        parts.append("\n# LAST_STEP (for context only, DO NOT output this JSON):\n" + commented_last_step + "\n")

    """
    try:
        c_str = json.dumps(constraints, ensure_ascii=False, indent=2)
    except Exception:
        c_str = str(constraints)
    if c_str.strip():
        if len(c_str) > 3000:
            c_str = c_str[:3000] + "\n# ... (truncated)"
        parts.append("\n# CONSTRAINTS (json)\n" + c_str.strip() + "\n")
    """


    # 6) 마지막 트리거 - 간결하게
    parts.append(
"\n# CONSTRAINTS:\n"
"# - The following statements have already been executed successfully.\n"
"# - DO NOT repeat them.\n"
"# - Generate ONLY the remaining steps needed to complete the task.\n"
"# - Write continuation statements WITHOUT extra leading indentation (start at body-level).\n"
"# - IMPORTANT: Follow the indentation, parameters, UTILITIES and Args(find_place_position, use_smart_placement, etc.) shown in [REFERENCE CODE (TASK LOGIC SPEC)].\n"

"\n# Continue from here (add the remaining steps):\n"
    )

    revise_prompt = "\n".join(parts).strip() + "\n"

    return revise_prompt


# =========================
# LLM Backend (Abstract)
# =========================

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> Tuple[str, int]:
        """Generate code from prompt. Returns (code, num_tokens)."""
        pass

    @abstractmethod
    def close(self):
        pass


# =========================
# Lower Backend (Qwen local)
# =========================

class LowerBackend(LLMBackend):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Coder-7B"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)

        print(f"[Lower] Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=bnb_cfg,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        print(f"[Lower] Model loaded successfully")

    def generate(self, prompt: str) -> Tuple[str, int]:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
        num_tokens = len(gen_ids)
        code = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return code, num_tokens

    def close(self):
        pass



# =========================
# Upper Backend (OpenAI API)
# =========================

class UpperBackend(LLMBackend):
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for upper backend")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"[Upper] Using OpenAI model: {model}")

    def _strip_markdown(self, text: str) -> str:
        """Remove markdown code blocks from response."""
        pattern = r'```(?:python)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _uncomment_code(self, text: str, indent: int = 4) -> str:
        """Remove leading '# ' from all lines if the entire output is commented."""
        lines = text.strip().split('\n')

        non_empty_lines = [l for l in lines if l.strip()]
        if not non_empty_lines:
            return text

        comment_lines = [l for l in non_empty_lines if l.strip().startswith('#')]
        comment_ratio = len(comment_lines) / len(non_empty_lines)

        if comment_ratio > 0.8:
            result_lines = []
            indent_str = ' ' * indent
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('# '):
                    content = stripped[2:]
                    if content:
                        result_lines.append(indent_str + content)
                    else:
                        result_lines.append('')
                elif stripped == '#':
                    result_lines.append('')
                elif stripped:
                    result_lines.append(indent_str + stripped)
                else:
                    result_lines.append('')
            return '\n'.join(result_lines)

        return text

    def generate(self, prompt: str) -> Tuple[str, int]:
        messages = [
            {"role": "system", "content": (
                "You are a robot code generator for Genesis simulator manipulation tasks. "
                "Output ONLY valid, executable Python code - NO COMMENTS AT ALL. "
                "CRITICAL RULES: "
                "1. Do NOT use markdown code blocks (```). "
                "2. Do NOT add explanations or comments before/after the code. "
                "3. NEVER output commented-out code (lines starting with #). ALL code must be executable. "
                "4. Do NOT use placeholders. Fill in ALL values with actual numbers and strings. "
                "5. Start directly with the code body (variable assignments, function calls, etc.). "
                "6. The prompt contains comments for context, but YOUR OUTPUT must be pure executable code WITHOUT any # comments. "
                "7. Use Genesis-specific skills like move_gripper_to, pick, place, activate_vacuum, etc."
            )},
            {"role": "user", "content": prompt}
        ]

        if self.model.startswith("gpt-5"):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                #temperature=0.0,
                #top_p=1.0,
                max_completion_tokens=1024,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                #temperature=0.0,
                #top_p=1.0,
                max_tokens=1024,
            )

        raw = response.choices[0].message.content
        code = self._strip_markdown(raw)
        code = self._uncomment_code(code)

        # Extract token count from response
        num_tokens = response.usage.completion_tokens if response.usage else 0

        return code, num_tokens

    def close(self):
        pass


# =========================
# Global Backend Instance
# =========================

_BACKEND: Optional[LLMBackend] = None
_BACKEND_TYPE: str = "none"


# =========================
# Parameter Filtering
# =========================

def filter_invalid_params(generated_code: str, robot_type: str = "panda") -> str:
    """
    Filter out parameters from generated function calls that are not in the skill signature.
    """
    robot_skills = GENESIS_SKILLS.get(robot_type, GENESIS_SKILLS["panda"])
    signatures = robot_skills["signatures"]

    if not signatures:
        return generated_code

    result_lines = []
    for line in generated_code.split('\n'):
        for func_name in signatures.keys():
            if func_name not in line:
                continue

            # Extract function call with balanced parentheses
            func_idx = line.find(func_name)
            if func_idx == -1:
                continue

            paren_start = line.find('(', func_idx)
            if paren_start == -1:
                continue

            # Find matching closing paren
            paren_depth = 1
            i = paren_start + 1
            while i < len(line) and paren_depth > 0:
                if line[i] == '(':
                    paren_depth += 1
                elif line[i] == ')':
                    paren_depth -= 1
                i += 1

            if paren_depth != 0:
                continue

            paren_end = i
            args_str = line[paren_start + 1:paren_end - 1]
            valid_params = set(signatures[func_name])

            # Parse keyword arguments with proper nesting handling
            filtered_args = []
            current_arg = ""
            paren_depth = 0
            bracket_depth = 0

            for char in args_str + ',':
                if char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == '[':
                    bracket_depth += 1
                    current_arg += char
                elif char == ']':
                    bracket_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                    arg = current_arg.strip()
                    if arg:
                        if '=' in arg:
                            param_name = arg.split('=')[0].strip()
                            if param_name in valid_params:
                                filtered_args.append(arg)
                            else:
                                print(f"[Filter] Removing invalid param '{param_name}' from {func_name}")
                        else:
                            filtered_args.append(arg)
                    current_arg = ""
                else:
                    current_arg += char

            new_args_str = ', '.join(filtered_args)
            new_call = f"{func_name}({new_args_str})"
            line = line[:func_idx] + new_call + line[paren_end:]

        result_lines.append(line)

    return '\n'.join(result_lines)


def init_backend(backend_type: str, model: str = None):
    global _BACKEND, _BACKEND_TYPE

    _BACKEND_TYPE = backend_type

    if backend_type == "lower":
        model_id = model or "Qwen/Qwen2.5-Coder-7B"
        _BACKEND = LowerBackend(model_id)
    elif backend_type == "upper":
        model_name = model or "gpt-4o"
        _BACKEND = UpperBackend(model_name)
    else:
        _BACKEND = None
        print(f"[WARN] Unknown backend type: {backend_type}, running in dummy mode")


# =========================
# Client handler
# =========================

def handle_client(conn, addr):
    try:
        print(f"[DEBUG] Client connected from {addr}")
        try:
            msg = recv_json(conn)
            print(f"[DEBUG] Received message: {msg}")
        except Exception as e:
            print(f"[ERROR] Failed to receive/parse JSON from {addr}: {e}")
            import traceback
            traceback.print_exc()
            return

        if msg.get("type") != "hello":
            print(f"[ERROR] Expected 'hello', got '{msg.get('type')}'. Closing connection.")
            return
        print(f"[DEBUG] Sending hello_ack to {addr}")
        send_json(conn, {"type": "hello_ack"})

        while True:
            req = recv_json(conn)
            rtype = req.get("type")
            print(f"{addr} -> {rtype}")

            # Generate(INITIAL) Handler
            if rtype == "generate":
                try:
                    robot_type = _normalize_robot_type(req.get("target_robot") or req.get("robot_type") or "panda")
                    # robot_type = req.get("robot_type", "panda")
                    num_tokens = 0

                    if _BACKEND is not None:
                        full_prompt = build_generate_prompt(req)
                        
                        print(f"[Generate] Robot: {robot_type}, Prompt length: {len(full_prompt)}")
                        if req.get("debug_prompt", False):
                            lines = full_prompt.splitlines()
                            print("\n[DEBUG][PROMPT][GENERATE]:\n" + "\n".join(lines))
                            print(f"\n[DEBUG][PROMPT][GENERATE] chars={len(full_prompt)}, lines={len(lines)}\n")

                        code, num_tokens = _BACKEND.generate(full_prompt)
                        print("[Generated code] \n", code)
                        # Filter invalid parameters
                        code = filter_invalid_params(code, robot_type)
                        print("[Filtered code] \n", code)
                    else:
                        # Dummy response for testing
                        code = "    pass  # Dummy response"

                    send_json(conn, {
                        "type": "generate_result",
                        "ok": True,
                        "code": code,
                        "num_tokens": num_tokens
                    })

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    send_json(conn, {
                        "type": "generate_result",
                        "ok": False,
                        "error": str(e),
                        "num_tokens": 0
                    })

            elif rtype == "revise":
                try:
                    robot_type = _normalize_robot_type(req.get("target_robot") or req.get("robot_type") or "panda")
                    num_tokens = 0

                    if _BACKEND is None:
                        raise RuntimeError("revise 요청을 처리할 LLM backend가 없습니다. (current_code fallback 비활성화됨)")

                    full_prompt = build_revise_prompt(req)

                    print(f"[REVISE] Robot: {robot_type}, Prompt length: {len(full_prompt)}")
                    if req.get("debug_prompt", False):
                        _debug_print_prompt(full_prompt, tag="REVISE", head_n=120, tail_n=80)

                    code, num_tokens = _BACKEND.generate(full_prompt)
                    print("[REVISED CODE] \n", code[:500], "..." if len(code) > 500 else "")
                    code = filter_invalid_params(code, robot_type)

                    send_json(conn, {
                        "type": "revise_result",
                        "ok": True,
                        "code": code,
                        "num_tokens": num_tokens
                    })

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    send_json(conn, {
                        "type": "revise_result",
                        "ok": False,
                        "error": str(e),
                        "num_tokens": 0
                    })


            elif rtype == "repair":
                try:
                    robot_type = _normalize_robot_type(req.get("target_robot") or req.get("robot_type") or "panda")
                    num_tokens = 0

                    if _BACKEND is None:
                        raise RuntimeError("repair 요청을 처리할 LLM backend가 없습니다. (current_code fallback 비활성화됨)")

                    full_prompt = build_repair_prompt(req)

                    print(f"[REPAIR] Robot: {robot_type}, Prompt length: {len(full_prompt)}")
                    if req.get("debug_prompt", False):
                        _debug_print_prompt(full_prompt, tag="REPAIR", head_n=120, tail_n=80)

                    code, num_tokens = _BACKEND.generate(full_prompt)
                    print("[REPAIRED CODE] \n", code[:500], "..." if len(code) > 500 else "")
                    code = filter_invalid_params(code, robot_type)

                    send_json(conn, {
                        "type": "repair_result",
                        "ok": True,
                        "code": code,
                        "num_tokens": num_tokens
                    })

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    send_json(conn, {
                        "type": "repair_result",
                        "ok": False,
                        "error": str(e),
                        "num_tokens": 0
                    })


            else:
                send_json(conn, {
                    "type": "error",
                    "ok": False,
                    "error": f"unknown type: {rtype}"
                })

    except Exception as e:
        print(f"{addr} disconnected: {e}")
    finally:
        conn.close()

# =========================
# Server loop
# =========================

def serve(port: int = 9000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen()
        print(f"[Genesis Server] Listening on 127.0.0.1:{port} (backend={_BACKEND_TYPE})")

        while True:
            conn, addr = s.accept()
            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            ).start()


# =========================
# Entry
# =========================

def main():
    global _USE_DEFAULT_LLM, _DEFAULT_SOURCE_ROBOT, _DEFAULT_TARGET_ROBOT

    parser = argparse.ArgumentParser(
        description="Inference Server for Genesis Robot Code Generation"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["lower", "upper", "none"],
        default="lower",
        help="LLM backend: lower (Qwen local), upper (OpenAI API), none (dummy)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='Qwen/Qwen2.5-Coder-7B',
        help="Model name/id (lower: 'Qwen/Qwen2.5-Coder-7B', upper: 'gpt-4o')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Port to listen on (default:9000)"
    )
    parser.add_argument(
        "--use-default-llm",
        action="store_true",
        help="Use default LLM for generate (fallback)"
    )
    parser.add_argument(
        "--source-robot", "-s",
        type=str,
        default="panda",
        choices=["panda"],
        help="Default source robot type (default: panda)"
    )
    parser.add_argument(
        "--target-robot", "-r", 
        type=str,
        default="robotiq",
        choices=["panda", "robotiq", "suction"],
        help="Default target robot type (default: robotiq)"
    )
    args = parser.parse_args()

    _USE_DEFAULT_LLM = args.use_default_llm
    _DEFAULT_SOURCE_ROBOT = args.source_robot
    _DEFAULT_TARGET_ROBOT = args.target_robot

    # if _USE_DEFAULT_LLM:
    #     init_default_llm()

    # Initialize infilling LLM at server start
    print(f"[Genesis Server] Source robot: {_DEFAULT_SOURCE_ROBOT}")
    print(f"[Genesis Server] Target robot: {_DEFAULT_TARGET_ROBOT}")
 
    if args.backend != "none":
        init_backend(args.backend, args.model)

    serve(args.port)


if __name__ == "__main__":
    main()



"""
def main():
    parser = argparse.ArgumentParser(description="Inference Server for Genesis Robot Code Generation")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["lower", "upper", "none"],
        default="lower",
        help="LLM backend: lower (Qwen local), upper (OpenAI API), none (dummy)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='Qwen/Qwen2.5-Coder-7B',
        help="Model name/id (lower: 'Qwen/Qwen2.5-Coder-7B', upper: 'gpt-4o')"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Server port (default: 9000)"
    )
    args = parser.parse_args()

    if args.backend != "none":
        init_backend(args.backend, args.model)

    serve(args.port)


if __name__ == "__main__":
    main()
"""

    # reference_section = f"\n# [REFERENCE CODE (TASK LOGIC SPEC)]\n{processed_ref}\n\n"
    # Find insertion point for reference code (before "# RULES")
    # rules_pos = prompt.find("# RULES")
    # if rules_pos != -1 and reference_section:
    #     prompt = prompt[:rules_pos] + reference_section + prompt[rules_pos:]