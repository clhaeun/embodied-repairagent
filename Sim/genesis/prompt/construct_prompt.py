"""
Prompt construction module for Genesis Simulator tasks.

This module constructs prompts for LLM-based code generation,
providing task-specific context and skill APIs.
"""

from typing import List, Optional


def _get_object_relation(task_name: str, objects: List[str], instruction: str) -> str:  # noqa: ARG001
    """
    Generate object relation description based on task type and objects.
    """
    # Filter out non-manipulable objects
    manipulable = [obj for obj in objects if obj not in ('floor', 'gripper', 'suction')]

    # Task-specific relation templates
    if task_name == "pick_place":
        containers = [obj for obj in manipulable if any(kw in obj.lower() for kw in ('box', 'bin', 'container', 'basket', 'tray'))]
        items = [obj for obj in manipulable if obj not in containers]

        if containers and items:
            container_str = containers[0]
            items_str = ", ".join(items)
            return f"The objects ({items_str}) are initially located outside the {container_str} and should be placed into the {container_str}."
        else:
            return f"The objects are scattered on the workspace and need to be rearranged according to the task."

    elif task_name == "put_in_hinge":
        hinged = [obj for obj in manipulable if any(kw in obj.lower() for kw in ('hinge', 'door', 'lid', 'cabinet'))]
        items = [obj for obj in manipulable if obj not in hinged]

        if hinged and items:
            hinged_obj = hinged[0]
            items_str = ", ".join(items)
            return f"The {hinged_obj} has a hinged lid/door. The objects ({items_str}) should be placed inside after opening the lid."
        else:
            return f"A hinged container needs to be opened before placing objects inside."

    elif task_name == "put_in_prismatic":
        prismatic = [obj for obj in manipulable if any(kw in obj.lower() for kw in ('drawer', 'slide', 'prismatic'))]
        items = [obj for obj in manipulable if obj not in prismatic]

        if prismatic and items:
            prismatic_obj = prismatic[0]
            items_str = ", ".join(items)
            return f"The {prismatic_obj} has a sliding drawer. The objects ({items_str}) should be placed inside after opening the drawer."
        else:
            return f"A sliding drawer needs to be opened before placing objects inside."

    elif task_name == "sweep":
        targets = [obj for obj in manipulable if any(kw in obj.lower() for kw in ('dust', 'debris', 'particle', 'crumb'))]

        if targets:
            targets_str = ", ".join(targets)
            return f"The objects ({targets_str}) need to be swept to a target location using pushing/sweeping motions."
        else:
            return f"Objects on the surface need to be swept/pushed to a designated area."

    elif task_name == "composite":
        return f"This is a composite task requiring multiple sub-tasks. Objects: {', '.join(manipulable)}."

    else:
        return f"Objects available for manipulation: {', '.join(manipulable)}."


def construct_prompt(
    task_name: str,
    instruction: str,
    objects: List[str],
    ee_type: str = "gripper",
    object_relation: Optional[str] = None,
) -> str:
    """
    Construct a prompt for LLM-based code generation.

    Args:
        task_name: Name of the task (e.g., "pick_place", "put_in_hinge")
        instruction: Task instruction/description (e.g., "Place the cube blocks in to the box.")
        objects: List of object names in the scene (e.g., ['floor', 'toy_box', 'red_cube', 'gripper'])
        ee_type: End-effector type ("gripper" or "suction")
        object_relation: Optional custom object relation description

    Returns:
        Formatted prompt string for LLM code generation
    """
    # Filter out 'floor' from available objects for manipulation
    available_objects = [obj for obj in objects if obj != 'floor']

    # Generate object relation if not provided
    if object_relation is None:
        object_relation = _get_object_relation(task_name, objects, instruction)

    # Create object list string
    obj_list_str = ", ".join([f"'{obj}'" for obj in available_objects])

    # Note: Vacuum EE instructions are already included in the prompt template below
    _ = ee_type  # ee_type can be used for future conditional prompt sections

    prompt = f"""
# MANDATORY: Use these EXACT objects and skills. NO placeholders!
# Available objects: {obj_list_str}
# Object relation: {object_relation}
# Available primitive skills: move_gripper_to, move_to_position, move_parallel, open_gripper, close_gripper, grasp_handle, release_handle, pick, place

# RULES (IMPORTANT)
# - Goal: produce pick_place.py-style control code:
#   loop over manipulable objects -> pick -> move to target -> release -> env.check_result().
# - Prefer HIGH-LEVEL composition:
#   - For simple rigid objects: pick(env, obj, ...) then move_to_position(...) then open_gripper(...)
#   - Or: move_gripper_to(...) + close_gripper(...) + move_to_position(...) + open_gripper(...)
# - move_gripper_to() already computes approach using bbox + depth; avoid redundant pre-moves unless needed.
# - pointing_to MUST be a STRING in {{'down','left','right'}} (no other values).
# - move_dir MUST be a STRING in {{'left','right','front','back','up'}} (no other values).
# - Use reasonable values:
#   - depth ~ -0.01 to 0.02
#   - lift_clearance ~ 0.08-0.15
#   - place height offset (+z) ~ 0.05-0.15
# - Always check visibility with env.is_obj_visible(name) before attempting to manipulate.
# - End by calling env.check_result() and return it.
# - Do NOT use hard-coded absolute world coordinates; always compute target positions relative to object positions (e.g., pos = env.get_obj_pos(name) + [dx, dy, dz]).

# Available primitive skills (reference)
# - move_gripper_to(env, obj_name, pointing_to=____, depth=____) -> bool
# - move_to_position(env, pos=[____,____,____], pointing_to=____, lift_clearance=____, angle=____) -> None
# - move_parallel(env, move_dir=____, offset=____, pointing_to=____) -> None
# - open_gripper(env) -> None
# - close_gripper(env) -> None
# - grasp_handle(env, handle_name) -> bool
# - release_handle(env) -> None
# - pick(env, obj_name, pointing_to=____) -> bool
# - place(env, obj_name, pointing_to=____) -> bool

# Perception APIs (reference)
# - env.get_obj_names() -> list[str]
# - env.is_obj_visible(obj_name) -> bool
# - env.get_obj_pos(obj_name) -> np.ndarray shape (3,)
# - env.get_obj_bbox(obj_name) -> np.ndarray shape (2,3) or None
# - env.get_obj_size(obj_name) -> np.ndarray shape (3,)
# - env.get_obj_angle(obj_name) -> float
# - env.get_obj_speed(obj_name) -> float
# - env.obj_in_gripper(obj_name) -> bool
# - env.get_empty_floor_xy(obj_name, grid_step=None, max_tries_sort_prefix=0) -> np.ndarray shape (2,)
# - env.check_result() -> result string or None

# OUTPUT: Only write valid Python code for run_skill. No markdown. No comments.

# Primitive call shapes (reference)
# ok = move_gripper_to(env, obj_name=____, pointing_to=____, depth=____)
# move_to_position(env, pos=[____,____,____], pointing_to=____, lift_clearance=____, angle=____)
# move_parallel(env, move_dir=____, offset=____, pointing_to=____)
# open_gripper(env)
# close_gripper(env)
# ok = grasp_handle(env, handle_name=____)
# release_handle(env)
# ok = pick(env, obj_name=____, pointing_to=____)
# ok = place(env, obj_name=____, pointing_to=____)
# (Vacuum EE only)
# activate_vacuum(env)
# deactivate_vacuum(env)
# attach_vacuum_handle(env, handle_name=____)
# detach_vacuum_handle(env)

# Composition patterns (reference)
# Example 1. pick then place into a container (by position)
# obj_pos = env.get_obj_pos(____)
# target_pos = env.get_obj_pos(____)
# ok = pick(env, obj_name=____, pointing_to=____)
# place_pos = target_pos.copy(); place_pos[2] += ____
# move_to_position(env, pos=place_pos.tolist(), pointing_to=____, lift_clearance=____, angle=____)
# open_gripper(env)  # or deactivate_vacuum(env) depending on ee_type
#
# Example 2. explicit approach + grasp + transfer + release
# ok = move_gripper_to(env, obj_name=____, pointing_to=____, depth=____)
# close_gripper(env)  # or activate_vacuum(env) + attach_vacuum_handle(env, handle_name=____)
# move_to_position(env, pos=[____,____,____], pointing_to=____, lift_clearance=____, angle=____)
# open_gripper(env)   # or detach_vacuum_handle(env, handle_name=____) + deactivate_vacuum(env)
#
# Example 3. handle interaction (if task uses handles)
# ok = move_gripper_to(env, obj_name=____, pointing_to=____, depth=____)
# ok = grasp_handle(env, handle_name=____)
# move_parallel(env, move_dir=____, offset=____, pointing_to=____)
# release_handle(env)

def run_skill(env, task, descriptions=None, obs=None, variation_index: int = 1):
    \"\"\"
    Main function to run the {task_name} task with the given environment and task.
    Task Description: {instruction}
    \"\"\"

    # First, get positions / names of the available objects
"""

    return prompt
