from textwrap import dedent


initial_prompt_dict = {
    "CODE_AGENT": {
        "panda": dedent("""\
            # Task: {task_name}
            # You MUST use the EXACT objects and skills provided. NO placeholders, NO new APIs.
            #
            ## Current state
            # GENERATE: While in this state, your objective is to write an initial complete skill implementation for the task using ONLY the provided objects and primitive skills. Produce direct, executable Python code that follows the task goal. Do not add placeholders or new primitive skills.
            #
            # Available objects: {obj_list_str}
            # Available primitive skills: move, pick, place, push, open_gripper, close_gripper, align_to_quaternion, align_two_axes
            # OUTPUT: Only write valid Python code for run_skill body. No markdown. No comments.

            # RULES (IMPORTANT)
            # - pick/place internally move to target_pos. pick() moves then grips; place() moves then releases.
            # - push includes close_gripper() followed by move() to the target position internally.
            # - approach_axis must be a STRING in {{'x','y','z','-x','-y','-z'}} (do NOT use AXIS.* or enums).
            # - Use reasonable values: approach_distance ~ 0.05-0.15, timeout ~ 3.0-10.0.
            # - align_to_quaternion is NOT a default step. Use it ONLY when top-down approach is not possible.
            # - offset is applied when computing target_pos: target_pos = object_pos + [dx, dy, dz]
            #   Example: target_pos = pos + [0.0, 0.0, 0.2]  # 20cm above
            # - yaw_align MUST be either 'parallel' or 'perpendicular' (no other values allowed)
            # - WORKSPACE LIMIT: Target positions MUST stay within the robot's reachable workspace (~0.8m radius from base).
            #   Do NOT add large offsets (> 0.15m in x/y) to object positions. If PathOutOfWorkspace occurs, use smaller offsets or positions closer to scene objects.

            # Primitive call shapes (reference)
            # pick:  obs, reward, done = pick(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
            # place: obs, reward, done = place(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
            # move:  obs, reward, done = move(env, task, target_pos=[x,y,z], timeout=FLOAT)  # rare; pick/place usually enough
            # push:  obs, reward, done = push(env, task, target_pos=[x,y,z], approach_axis='z', approach_distance=FLOAT, timeout=FLOAT)
            # grip:  obs, reward, done = open_gripper(env, task) / close_gripper(env, task)
            # align:
            #   q = obj.get_quaternion()
            #   obs, reward, done = align_to_quaternion(env, task, quaternion=q, yaw_align='parallel', approach_dir=APPROACH_DIR, timeout=FLOAT)
            # align_two_axes:
            #   obs, reward, done = align_two_axes(env, task, local_axes=('x','y'), world_axes=('z','y'), axis_dirs=(1,1), timeout=FLOAT)

            # Utility functions available (reference):
            # - get_bbox_sizes(obj): returns (sizes, bbox) where sizes=[dx,dy,dz]
            # - quat_mul(q1, q2): quaternion multiplication
            # - normalize_vector(v): normalize vector with error handling

            # Composition patterns (reference)
            # Example 1. pick then place
            # obj = Shape('object_name')
            # pos = obj.get_position()
            # target_pos = pos + [0.0, 0.0, 0.2]
            # obs, reward, done = pick(env, task, target_pos=pos, approach_axis=____, approach_distance=____, timeout=____)
            # obs, reward, done = place(env, task, target_pos=target_pos, approach_axis=____, approach_distance=____, timeout=____)
            #
            # Example 2. align then pick/place
            # q = obj.get_quaternion()
            # obs, reward, done = align_to_quaternion(env, task, quaternion=q, yaw_align=____, approach_dir=____, timeout=____)


            def run_skill(env, task, descriptions=None, obs=None, variation_index: int = 1):
                \"\"\"
                Main function to run the {task_name} task with the given environment and task.
                Task Description: {descriptions}
                \"\"\"

                # CRITICAL: Use Shape('object_name') or Joint('object_name') to access objects. NEVER use env.scene
                # Example: obj = Shape('target_object'); pos = obj.get_position()
        """),
        "ur5": dedent("""
            # Task: {task_name}
            # You MUST use the EXACT objects and skills provided. NO placeholders, NO new APIs.
            #
            ## Current state
            # GENERATE: While in this state, your objective is to write an initial complete skill implementation for the task using ONLY the provided objects and primitive skills. Produce direct, executable Python code that follows the task goal. Do not add placeholders or new primitive skills.
            #
            # Available objects: {obj_list_str}
            # Available primitive skills: ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper, close_ur5_ee, open_ur5_ee
            # OUTPUT: Only write valid Python code for run_skill. No markdown. No comments.

            # RULES (IMPORTANT)
            # - DO NOT use if done: / else branching or other control-flow constructs (e.g., while loops).
            # - Every primitive skill call MUST be written as: obs, reward, done = <skill>(...)
            # - Never access env.scene or env.scene.objects (it does not exist). Use documented Shape and Joint access patterns.
            # - ur5_grasp_at/ur5_release_at already include moving/approach internally. Do NOT add extra ur5_move_to() unless truly needed.
            # - approach must be a DICT with 'axis' (STRING in {{'x','y','z','-x','-y','-z'}}) and 'distance' (FLOAT) (do NOT use AXIS.* or enums).
            # - Unless there is a specific reason, use only 'z' or '-z' for approach['axis'] (avoid 'x'/'y' approaches) for ur5_grasp_at.
            # - If "GRASP_GUIDANCE" is provided in the scene/error payload, implement it in code exactly
            # - Before calling ur5_align_gripper, first move the gripper to a safe pre-grasp pose (e.g., above_grasp_pos) using ur5_move_to().
            # - For push-related tasks, call close_ur5_ee() before any push/press motion.
            # - Use a longer timeout_s than 5.0 when needed (e.g., 8.0-15.0) to allow sufficient time for waypoint motion and avoid premature failures.
            # - Use reasonable values: approach distance ~ 0.05-0.15, timeout_s ~ 8.0-15.0.
            # - ur5_align_gripper is NOT a default step. Use it ONLY when top-down approach is not possible.
            # - offset is applied when computing target_pos: target_pos = object_pos + [dx, dy, dz]
            #   Example: target_pos = pos + [0.0, 0.0, 0.2]  # 20cm above
            # - yaw_mode MUST be either 'parallel' or 'perpendicular' (no other values allowed)
            # - WORKSPACE LIMIT: Target positions MUST stay within the robot's reachable workspace (~0.8m radius from base).
            #   Do NOT add large offsets (> 0.15m in x/y) to object positions. If PathOutOfWorkspace occurs, use smaller offsets or positions closer to scene objects.

            # Primitive call shapes (reference)
            # ur5_grasp_at:  obs, reward, done = ur5_grasp_at(env, task, grasp_pos=[x,y,z], approach={{'axis':'z', 'distance':FLOAT}}, timeout_s=FLOAT)
            # ur5_release_at: obs, reward, done = ur5_release_at(env, task, place_pos=[x,y,z], approach={{'axis':'z', 'distance':FLOAT}}, timeout_s=FLOAT)
            # ur5_move_to:  obs, reward, done = ur5_move_to(env, task, target_pos=[x,y,z], timeout_s=FLOAT)  # rare; ur5_grasp_at/ur5_release_at usually enough
            # ur5_align_gripper:
            #   q = obj.get_quaternion()
            #   obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction=APPROACH_DIR, timeout_s=FLOAT)
            # close_ur5_ee: obs, reward, done = close_ur5_ee(env, task, gripper_close=0.0, velocity=0.2)  # close gripper (0.0=closed, 1.0=open)
            # open_ur5_ee:  obs, reward, done = open_ur5_ee(env, task, gripper_open=1.0, velocity=0.2)   # open gripper (0.0=closed, 1.0=open)

            # Composition patterns (MUST FOLLOW)
            # Example 1. ur5_align_gripper, then ur5_grasp_at, then ur5_release_at
            # When GRASP_GUIDANCE is provided, always align before grasping:
            # obj = Shape('object_name')
            # obj_pos = obj.get_position()
            #
            # Define the target positions for pick and place
            # pick_pos = obj_pos + [0.0, 0.0, 0.1]
            #
            # Move to the pick position
            # obs, reward, done = ur5_move_to(env, task, target_pos=pick_pos, timeout_s=10.0)
            #
            # Align the gripper to the obj
            # q = obj.get_quaternion()
            # obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, approach_direction="down", yaw_mode="parallel", timeout_s=5.0)
            #
            # Grasp the obj
            # obs, reward, done = ur5_grasp_at(env, task, grasp_pos=pos, timeout_s=5.0)

            # Example 2. ur5_grasp_at then ur5_release_at
            # ball = Shape('ball')
            # target = Shape('target_respondable')
            # pos = ball.get_position()
            # target_pos = target.get_position()
            # target_pos = target_pos + [0.0, 0.0, 0.15]
            # obs, reward, done = ur5_grasp_at(env, task, grasp_pos=pos, timeout_s=5.0)
            # obs, reward, done = ur5_release_at(env, task, place_pos=target_pos, timeout_s=5.0)

        def run_skill(env, task, descriptions=None, obs=None, variation_index: int = 1):
            \"\"\"
            Main function to run the {task_name} task with the given environment and task.
            Task Description: {descriptions}
            Available UR5 PRIMITIVES: ur5_grasp_at, ur5_release_at, ur5_move_to, ur5_align_gripper, close_ur5_ee, open_ur5_ee
            \"\"\"
            # IMPORTANT: You MUST ONLY use UR5 PRIMITIVES and UTILITIES strictly.

            # CRITICAL: Use Shape('object_name') or Joint('object_name') to access objects. NEVER use env.scene
            # Example: obj = Shape('target_object'); pos = obj.get_position()
        """),
        "sawyer": dedent("""
            # Task: {task_name}
            # You MUST use the EXACT objects and skills provided. NO placeholders, NO new APIs.
            #
            ## Current state
            # GENERATE: While in this state, your objective is to write an initial complete skill implementation for the task using ONLY the provided objects and primitive skills. Produce direct, executable Python code that follows the task goal. Do not add placeholders or new primitive skills.
            #
            # Available objects: {obj_list_str}
            # Available primitive skills: sawyer_move_to, sawyer_pick, sawyer_place, sawyer_align_gripper, sawyer_open_gripper, sawyer_close_gripper
            # OUTPUT: Only write valid Python code for run_skill. No markdown. No comments.

            # RULES (IMPORTANT)
            # - DO NOT use if done: / else branching or other control-flow constructs (e.g., while loops).
            # - Every primitive skill call MUST be written as: obs, reward, done = <skill>(...)
            # - Never access env.scene or env.scene.objects (it does not exist).
            # - sawyer_pick/sawyer_place already include moving and aligning internally. sawyer_pick() moves then grips; sawyer_place() moves then releases.
            # - sawyer_pick: target_object is REQUIRED. target_pos is OPTIONAL (defaults to object position if None).
            # - sawyer_place: place_pos is REQUIRED. No target_object parameter.
            # - Use reasonable values: timeout_s ~ 30.0-60.0.
            # - sawyer_align_gripper is NOT A DEFAULT STEP. DO NOT USE it.
            # - offset is applied when computing target_pos: target_pos = object_pos + [dx, dy, dz]
            #   Example: target_pos = pos + [0.0, 0.0, 0.2]  # 20cm above
            # - yaw_mode MUST be either 'parallel' or 'perpendicular' (no other values allowed)
            # - WORKSPACE LIMIT: Target positions MUST stay within the robot's reachable workspace (~0.8m radius from base).
            #   Do NOT add large offsets (> 0.15m in x/y) to object positions. If PathOutOfWorkspace occurs, use smaller offsets or positions closer to scene objects.

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
            # Example 1. pick then place (do not use sawyer_align_gripper)
            # obj = Shape('object_name'); pos = obj.get_position()
            # obs, reward, done = sawyer_pick(env, task, target_object=obj, target_pos=pos, grasp_offset=np.array([0.0, 0.0, 0.0]))
            # obs, reward, done = sawyer_place(env, task, place_pos=above_place_pos)

            # OUTPUT FORMAT (CRITICAL)
            # - DO NOT output "def run_skill" or any function/class definition.
            # - Output ONLY the indented function body to go inside run_skill (each line must start with 4 spaces).
            # - Your first output line MUST start with 4 spaces and be an executable statement (not empty).
            # - No markdown, no explanations.

        def run_skill(env, task, descriptions=None, obs=None, variation_index: int = 1):
            \"\"\"
            Main function to run the {task_name} task with the given environment and task.
            Task Description: {descriptions}
            \"\"\"

            # CRITICAL: Use Shape('object_name') or Joint('object_name') to access objects. NEVER use env.scene
            # Example: obj = Shape('lamp_screw'); pos = obj.get_position()
        """),
    }
}





































def _quote_obj_list(object_names):

    return ", ".join(f"'{name}'" for name in object_names)

def get_prompt(task, task_name, object_names, target_robot, descriptions, model, grasp_guidance: str = ""):
    model_key = model.upper()

    if model_key == 'OURS':
        return "Temp"

    if model_key not in initial_prompt_dict:
        raise ValueError(f"Unknown model type: {model}")

    if target_robot not in initial_prompt_dict[model_key]:
        raise ValueError(f"Unknown target_robot: {target_robot}")

    base_template = initial_prompt_dict[model_key][target_robot]


    obj_list_str = _quote_obj_list(object_names)


    descriptions = descriptions or ""


    prompt = base_template.format(
        obj_list_str=obj_list_str,
        task_name=task_name,
        descriptions=descriptions,
    )

    return prompt


import json
from typing import Any, Dict, List, Optional

def _indent_to_body(stmt_texts: List[str], indent: str = "    ") -> str:
    out = []
    for s in stmt_texts or []:
        s = str(s).rstrip("\n")
        if not s.strip():
            out.append("")
            continue

        if s.startswith((" ", "\t")):
            out.append(s)
        else:
            out.append(indent + s)
    return "\n".join(out)

def get_regenerate_prompt_on_failure(
    base_initial_prompt: str,
    task_name: str,
    robot_type: str,
    current_code: str,
    executed_stmt_texts: List[str],
    failed_stmt_text: str,
    error_payload: Dict[str, Any],
    scene: Dict[str, Any],
    failure_report: Optional[str] = None,
) -> str:
    executed_block = _indent_to_body(executed_stmt_texts, indent="    ")

    header = dedent(f"""
    [MODE]: REPAIR
    [TASK]: {task_name}
    [ROBOT]: {robot_type}

    [OUTPUT RULES - MUST FOLLOW]
    - Prefer linear code. Avoid if/for/while/try unless absolutely necessary.
    - Output ONLY the Python statements that continue INSIDE run_skill (function body continuation).
    - DO NOT output "def run_skill".
    - DO NOT repeat the provided prefix code.
    - NO markdown fences (no ```).
    - NO comments.
    - Keep indentation correct for being inside the function (4 spaces).
    - If you write a line ending with ":" (if/for/while/try/with), the very next line MUST be indented 8 spaces (4 more), and the whole block must be valid.
        Example:
            if cond:
                do_something()
            else:
                do_other()
    - You MUST NOT modify the already-executed prefix statements. Treat them as frozen.
    - Do NOT output incomplete blocks.

    """).strip()

    parts = [header]

    if failure_report:
        parts.append("[FAILURE_REPORT]\n" + failure_report.strip())

    parts.append("[FAILED_STATEMENT]\n" + (failed_stmt_text or "").strip())
    parts.append("[RUNTIME_ERROR]\n" + json.dumps(error_payload or {}, ensure_ascii=False))
    parts.append("[SCENE]\n" + json.dumps(scene or {}, ensure_ascii=False))
    parts.append("[CURRENT_CODE]\n" + (current_code or "").strip())



    prompt = "\n\n".join(parts) + "\n\n" + base_initial_prompt.rstrip() + "\n\n"
    if executed_block.strip():
        prompt += executed_block + "\n"
    prompt += "\n[CONTINUE FROM NEXT LINE NOW]\n"
    return prompt


def get_regenerate_prompt_on_not_finished(
    base_initial_prompt: str,
    task_name: str,
    robot_type: str,
    current_code: str,
    executed_stmt_texts: List[str],
    scene: Dict[str, Any],
    failure_report: Optional[str] = None,
) -> str:
    executed_block = _indent_to_body(executed_stmt_texts, indent="    ")

    header = dedent(f"""
    [MODE]: REPAIR (NOT_FINISHED)
    [TASK]: {task_name}
    [ROBOT]: {robot_type}

    [OUTPUT RULES - MUST FOLLOW]
    - Prefer linear code. Avoid if/for/while/try unless absolutely necessary.
    - Output ONLY the Python statements that continue INSIDE run_skill (function body continuation).
    - DO NOT output "def run_skill".
    - DO NOT repeat the provided prefix code.
    - NO markdown fences (no ```).
    - NO comments.
    - Keep indentation correct for being inside the function (4 spaces).
    - If you write a line ending with ":" (if/for/while/try/with), the very next line MUST be indented 8 spaces (4 more), and the whole block must be valid.
        Example:
            if cond:
                do_something()
            else:
                do_other()
    - You MUST NOT modify the already-executed prefix statements. Treat them as frozen.
    - Do NOT output incomplete blocks.

    """).strip()

    parts = [header]
    if failure_report:
        parts.append("[FAILURE_REPORT]\n" + failure_report.strip())
    parts.append("[SCENE]\n" + json.dumps(scene or {}, ensure_ascii=False))
    parts.append("[CURRENT_CODE]\n" + (current_code or "").strip())

    prompt = "\n\n".join(parts) + "\n\n" + base_initial_prompt.rstrip() + "\n\n"
    if executed_block.strip():
        prompt += executed_block + "\n"
    prompt += "\n[CONTINUE FROM NEXT LINE NOW]\n"
    return prompt
















