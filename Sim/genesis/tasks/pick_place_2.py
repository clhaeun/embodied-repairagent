"""
Pick and Place 2 Task Execution (BasePickPlaceTask variants).

Task: Place target objects (ball/cube/cylinder) into the tray while avoiding distractors.
Supports both normal tray and inverse tray (flipped, no walls).

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.pick_place_2 import check_pick_and_place_success, check_lid_open
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the pick and place 2 task.

    This task involves:
    1. Opening the box lid
    2. Identifying target objects (by target_type)
    3. Picking up each target object
    4. Placing them into the tray (avoiding distractors)

    Works for all variants:
    - pick_place_ball, pick_place_cube, pick_place_cylinder
    - pick_place_inverse_ball, pick_place_inverse_cube, pick_place_inverse_cylinder

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing target_names, distractor_names, and scene_objects.
        provider: PolicyProvider instance for subtask generation/execution.
        subtask_results: List to collect SubtaskResult.
        max_repairs: Max repair attempts per subtask.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        result: Task result ("full_success" or None).
    """
    if subtask_results is None:
        subtask_results = []

    print(f"[PickPlace2] Starting task")
    print(f"[PickPlace2] Target type: {task.target_type}")
    print(f"[PickPlace2] Target objects: {task.target_names}")
    print(f"[PickPlace2] Distractor objects: {task.distractor_names}")
    print(f"[PickPlace2] Inverse mode: {task.inverse}")


    print(f"\n[PickPlace2] Opening box lid first")
    try:
        result = provider.execute_subtask(
            subtask_name="open_box_lid",
            env=env,
            obj_name="box_handle",
            target_name="box_lid",
            check_func=lambda e, o, t: check_lid_open(e, "box_lid"),
            max_repairs=max_repairs,
            _subtask_results=subtask_results,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print(f"[PickPlace2] Skipping open_box_lid")
        if skip.result is not None:
            subtask_results.append(skip.result)


    from skill_code import move_parallel
    print(f"\n[PickPlace2] Moving gripper up and left to clear box")
    move_parallel(env, "up", 0.2, pointing_to="down")
    move_parallel(env, "left", 0.1, pointing_to="down")
    move_parallel(env, "down", 0.2, pointing_to="down")


    tray_pos = env.get_obj_pos("tray")
    print(f"[PickPlace2] Tray position: {tray_pos}")


    place_height = 0.05


    placed_objects = []


    for idx, obj_name in enumerate(task.target_names):
        print(f"\n[PickPlace2] Processing target {idx + 1}/{len(task.target_names)}: {obj_name}")


        spread_offset = (idx - (len(task.target_names) - 1) / 2) * 0.05

        try:
            result = provider.execute_subtask(
                subtask_name="pick_and_place_object",
                env=env,
                obj_name=obj_name,
                target_name="tray",
                check_func=lambda e, o, t: check_pick_and_place_success(e, o, t),
                max_repairs=max_repairs,
                target_pos=tray_pos,
                place_height=place_height,
                spread_offset=spread_offset,
                placed_objects=placed_objects,
                use_smart_placement=task.inverse,
                _subtask_results=subtask_results,
            )
            subtask_results.append(result)
            placed_objects.append(obj_name)
        except SubtaskSkip as skip:
            print(f"[PickPlace2] Skipping pick_and_place for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue


    result = env.check_result()
    print(f"\n[PickPlace2] Task completed with result: {result}")
    return result
