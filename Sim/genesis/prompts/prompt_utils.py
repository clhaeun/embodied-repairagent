"""
Prompt utilities for Genesis Experiment.

Provides prompt templates for code generation based on robot type.
"""

from textwrap import dedent



SUBTASK_SIGNATURES = {
    "close_drawer": (
        'def close_drawer(env, drawer_type="bottom"):',
        "Closes a prismatic drawer by grasping and pushing the handle."
    ),
    "close_hinge_lid": (
        'def close_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):',
        "Closes the lid of a hinge box by grasping and pushing the handle."
    ),
    "insert_into_slot": (
        'def insert_into_slot(env, obj_name, slot_pos=None, target_name=None, insert_height=0.05, **kwargs):',
        "Pick up an object and insert it into a slot."
    ),
    "open_box_lid": (
        'def open_box_lid(env, handle_name="box_handle", lid_name="box_lid"):',
        "Open the box lid. Use handle_name parameter directly (e.g., env.get_obj_pos(handle_name))."
    ),
    "open_drawer": (
        'def open_drawer(env, drawer_type="bottom"):',
        "Open a prismatic drawer by grasping and pulling the handle."
    ),
    "open_hinge_lid": (
        'def open_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):',
        "Open the hinge lid. Use handle_name parameter directly (e.g., env.get_obj_pos(handle_name))."
    ),
    "pick_and_place_object": (
        'def pick_and_place_object(env, obj_name, target_pos, place_height=0.01, spread_offset=0.0, placed_objects=None, use_smart_placement=True):',
        "Pick up ONLY the obj_name parameter (e.g., move_gripper_to(env, obj_name)) and place it at target_pos. Do NOT pick other objects."
    ),
    "pick_single": (
        'def pick_single(env, obj_name, pointing_to="down"):',
        "Pick up obj_name and lift it. Do NOT use target_name (no placing, only pick and lift)."
    ),
    "place_beside_slot": (
        'def place_beside_slot(env, obj_name=None, target_name=None, slot_pos=None, offset_x=-0.08, **kwargs):',
        "Place a held object beside a slot."
    ),
    "place_single": (
        'def place_single(env, target_name, pointing_to="down", height_offset=0.1):',
        "Place a held object at a target location."
    ),
    "place_at_position": (
        'def place_at_position(env, position, pointing_to="down"):',
        "Place a held object at a specific position."
    ),
    "grasp_container": (
        'def grasp_container(env, obj_name, pointing_to="down"):',
        "Grasp a container using gripper."
    ),
    "grasp_container_vacuum": (
        'def grasp_container_vacuum(env, obj_name, pointing_to="down"):',
        "Grasp a container using vacuum gripper. Use move_gripper_to + activate_vacuum."
    ),
    "pour_into_container": (
        'def pour_into_container(env, target_pos, pour_angle=120, pour_steps=60, hold_steps=100):',
        "Pour liquid into a target container."
    ),
    "release_container": (
        'def release_container(env, place_pos=None, original_pos=None, pointing_to="down", **kwargs):',
        "Release a container held by gripper."
    ),
    "release_container_vacuum": (
        'def release_container_vacuum(env, original_pos, pointing_to="down"):',
        "Release a container held by vacuum gripper. Use move_to_position + deactivate_vacuum."
    ),
    "relocate_single": (
        'def relocate_single(env, obj_name, target_name, pointing_to="down", height_offset=0.08, target_position=None):',
        "Relocate a misplaced object to its correct location. This subtask is used for objects that are in the wrong place (e.g., chess piece in wrong box) and need to be moved to the correct location."
    ),
    "remove_from_slot": (
        'def remove_from_slot(env, obj_name, lift_height=0.15, **kwargs):',
        "Pick up an object from a slot and lift it."
    ),
    "sweep_single": (
        'def sweep_single(env, obj_name, target_name, pointing_to="down"):',
        "Move a chess piece into a target box using pick-and-place."
    ),
    "unstack_single": (
        'def unstack_single(env, obj_name, pointing_to="down"):',
        "Remove the top object from a stack and place it aside."
    ),
}






initial_prompt_dict = {
    "CODE_AGENT": {
        "panda": dedent("""\
            # SUBTASK CODE GENERATION (Panda parallel jaw gripper)
            # Generate a SINGLE subtask function that operates on ONE object.
            # Available primitives: move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, grasp_handle, release_handle

            # RULES
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
            #     move_parallel(env, "up", 0.05, pointing_to="down")
            #     move_parallel(env, "front", 0.05, pointing_to="down")
            # release_handle(env)

            # EXAMPLE 3: Open drawer (side approach)
            # move_gripper_to(env, handle_name, pointing_to="right")
            # grasp_handle(env, handle_name)
            # move_parallel(env, "left", 0.12, pointing_to="right")
            # release_handle(env)

            def run_task(env, task, descriptions=None, obs=None):
                \"\"\"
                Subtask: {descriptions}
                \"\"\"
                # Available primitives: move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, grasp_handle, release_handle
                # IMPORTANT: Preserve the exact indentation (if/for block depth) from the reference code. Variables defined inside an if-block must only be used inside that same block.

                # First, get positions of the available objects
        """),
        "robotiq": dedent("""\
            # SUBTASK CODE GENERATION (Robotiq85 gripper)
            # Subtask: {descriptions}
            # Generate a SINGLE subtask function that operates on ONE object.
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

            def run_task(env, task, descriptions=None, obs=None):
                \"\"\"
                ONLY USE the Args given. Do not use other objects or parameters.
                Available ROBOTIQ PRIMITIVES: move_gripper_to, move_to_position, move_parallel, open_robotiq85, close_robotiq85, grasp_handle_robotiq85, release_handle_robotiq85
                \"\"\"
                # IMPORTANT: You MUST ONLY use ROBOTIQ PRIMITIVES and UTILITIES strictly.

                # First, get positions of the needed objects
        """),
        "suction": dedent("""\
            # SUBTASK CODE GENERATION (Suction/Vacuum gripper)
            # Subtask: {descriptions}
            # Generate a SINGLE subtask function that operates on ONE object.
            # Available SUCTION PRIMITIVES: move_gripper_to, move_to_position, move_parallel, activate_vacuum, deactivate_vacuum, attach_vacuum_handle, detach_vacuum_handle, rotate_gripper

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


            def run_task(env, task, descriptions=None, obs=None):
                \"\"\"
                ONLY USE the Args given. NEVER use other objects or parameters.
                If given 'position' from Args, strictly use 'position' not 'target_name'.
                ONLY Use activate_vacuum(env) + attach_vacuum_handle(env, handle_name) instead of grasp_handle.
                ONLY Use detach_vacuum_handle(env) + deactivate_vacuum(env) instead of release_handle.
                ONLY Use deactivate_vacuum(env) instead of open_gripper.
                ONLY Use activate_vacuum(env) instead of close_gripper.
                Available SUCTION PRIMITIVES: move_gripper_to, move_to_position, move_parallel, activate_vacuum, deactivate_vacuum, attach_vacuum_handle, detach_vacuum_handle, rotate_gripper
                \"\"\"
                # IMPORTANT: You MUST ONLY use SUCTION PRIMITIVES and UTILITIES strictly.

                # First, get positions of the needed objects
        """),
    },
    "OURS": {
        "panda": "Temp",
        "robotiq": "Temp",
        "suction": "Temp",
    }
}


def _quote_obj_list(object_names):
    """Format object names as quoted list string."""
    return ", ".join(f"'{name}'" for name in object_names)


def get_prompt(
    task,
    task_name: str,
    object_names: list,
    target_robot: str,
    descriptions: str,
    model: str,
    grasp_guidance: str = "",
    subtask_name: str = None
) -> str:
    """
    Generate prompt for code generation.

    Args:
        task: Task instance
        task_name: Name of the task
        object_names: List of available object names
        target_robot: Target robot type ("panda", "robotiq", or "suction")
        descriptions: Task description
        model: Model type ("code_agent" or "ours")
        grasp_guidance: Optional grasp guidance text
        subtask_name: Name of the subtask (for signature lookup)

    Returns:
        Formatted prompt string
    """
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


    subtask_signature = "def run_task(env, task, descriptions=None, obs=None):"
    if subtask_name and subtask_name in SUBTASK_SIGNATURES:
        subtask_signature = SUBTASK_SIGNATURES[subtask_name][0]

    prompt = base_template.format(
        obj_list_str=obj_list_str,
        task_name=task_name,
        descriptions=descriptions,
        subtask_signature=subtask_signature,
    )




    return prompt


def _get_skills_hint(target_robot: str) -> str:
    """
    Get available skills hint based on robot type.

    Args:
        target_robot: Robot type ("panda", "robotiq", or "suction")

    Returns:
        Skills hint string
    """
    if target_robot == "panda":
        return "move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, pick, place, rotate_gripper, grasp_handle, release_handle"
    elif target_robot == "robotiq":
        return "move_gripper_to, move_to_position, move_parallel, open_robotiq85, close_robotiq85, grasp_handle_robotiq85, release_handle_robotiq85"
    else:
        return "move_gripper_to, move_to_position, move_parallel, activate_vacuum, deactivate_vacuum, attach_vacuum_handle, detach_vacuum_handle, rotate_gripper"


def get_regenerate_prompt_on_failure(
    target_robot: str,
    current_code: str,
    executed_stmts: list,
    failed_stmt: str,
    error_info: dict,
    scene_info: dict,
    grasp_guidance: str = ""
) -> str:
    """
    Get prompt for code regeneration after execution failure.

    Args:
        target_robot: Target robot type ("panda", "robotiq", or "suction")
        current_code: Current code that failed
        executed_stmts: List of already executed statement texts
        failed_stmt: The statement that failed
        error_info: Error information dict with 'error_type' and 'error_message'
        scene_info: Scene information dict with 'gripper' and 'objects'
        grasp_guidance: Optional grasp guidance text

    Returns:
        Formatted repair prompt string
    """
    error_type = error_info.get("error_type", "Unknown")
    error_msg = error_info.get("error_message", "")


    gripper = scene_info.get("gripper", {})
    gripper_pos = gripper.get("position", [])
    gripper_open = gripper.get("is_open", True)

    objects = scene_info.get("objects", [])
    obj_lines = []
    for obj in objects[:10]:
        name = obj.get("name", "")
        pos = obj.get("position", [])
        if name and pos:
            obj_lines.append(f"#   {name}: {[round(p, 3) for p in pos]}")


    skills_hint = _get_skills_hint(target_robot)

    prompt = f"""# [REPAIR TASK] Fix the failed statement and continue.
# Available skills: {skills_hint}
#
# [ERROR] {error_type}: {error_msg}
#
# [FAILED STATEMENT]
# {failed_stmt}
#
# [EXECUTED STATEMENTS - DO NOT MODIFY]
# Count: {len(executed_stmts)}
"""

    if executed_stmts:
        prompt += "# --- executed code ---\n"
        for stmt in executed_stmts:
            for line in stmt.split('\n'):
                prompt += f"# {line}\n"
        prompt += "# --- end executed ---\n"

    prompt += f"""#
# [CURRENT SCENE]
# gripper_pos: {[round(p, 3) for p in gripper_pos] if gripper_pos else 'N/A'}
# gripper_open: {gripper_open}
"""

    if obj_lines:
        prompt += "# objects:\n" + "\n".join(obj_lines) + "\n"

    if grasp_guidance:
        prompt += f"#\n# [GRASP GUIDANCE] {grasp_guidance}\n"

    prompt += f"""#
# [CURRENT CODE]
{current_code}

# [INSTRUCTION]
# 1. Keep all executed statements unchanged
# 2. Fix the failed statement or replace it
# 3. Output the complete corrected run_task function body
# 4. No markdown, only valid Python code
"""

    return prompt


def get_regenerate_prompt_on_not_finished(
    target_robot: str,
    current_code: str,
    executed_stmts: list,
    scene_info: dict,
    last_step_info: dict = None,
    grasp_guidance: str = ""
) -> str:
    """
    Get prompt for code regeneration when task is not finished.

    Args:
        target_robot: Target robot type ("panda", "robotiq", or "suction")
        current_code: Current code that did not complete the task
        executed_stmts: List of already executed statement texts
        scene_info: Scene information dict with 'gripper' and 'objects'
        last_step_info: Optional info about last executed step
        grasp_guidance: Optional grasp guidance text

    Returns:
        Formatted revise prompt string
    """

    step_idx = 0
    skill_name = ""
    success = False
    if last_step_info:
        step_idx = last_step_info.get("step_index", 0)
        skill_name = last_step_info.get("skill_name", "")
        success = last_step_info.get("success", False)


    gripper = scene_info.get("gripper", {})
    gripper_pos = gripper.get("position", [])
    gripper_open = gripper.get("is_open", True)

    objects = scene_info.get("objects", [])
    obj_lines = []
    for obj in objects[:10]:
        name = obj.get("name", "")
        pos = obj.get("position", [])
        if name and pos:
            obj_lines.append(f"#   {name}: {[round(p, 3) for p in pos]}")


    skills_hint = _get_skills_hint(target_robot)

    prompt = f"""# [REVISE TASK] Task not finished - add more steps to complete.
# Available skills: {skills_hint}
#
# [STATUS] Task incomplete - need more steps!
"""

    if last_step_info:
        prompt += f"# [LAST STEP] index={step_idx}, skill={skill_name}, success={success}\n"

    prompt += f"""#
# [EXECUTED STATEMENTS - DO NOT MODIFY]
# Count: {len(executed_stmts)}
"""

    if executed_stmts:
        prompt += "# --- executed code ---\n"
        for stmt in executed_stmts:
            for line in stmt.split('\n'):
                prompt += f"# {line}\n"
        prompt += "# --- end executed ---\n"

    prompt += f"""#
# [CURRENT SCENE]
# gripper_pos: {[round(p, 3) for p in gripper_pos] if gripper_pos else 'N/A'}
# gripper_open: {gripper_open}
"""

    if obj_lines:
        prompt += "# objects:\n" + "\n".join(obj_lines) + "\n"

    if grasp_guidance:
        prompt += f"#\n# [GRASP GUIDANCE] {grasp_guidance}\n"

    prompt += f"""#
# [CURRENT CODE]
{current_code}

# [INSTRUCTION]
# 1. Keep all executed statements unchanged
# 2. Review remaining statements for correctness
# 3. Add more steps to complete the task
# 4. Output the complete run_task function body
# 5. No markdown, only valid Python code
"""

    return prompt
