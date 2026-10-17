"""
Put in Hinge Box Task Execution.

Task: Put fruits into the hinge box.
Handles obstacles: moves fruit off lid, opens closed lid.

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.put_in_hinge import (
    check_pick_success,
    check_place_success,
    check_open_hinge_success,
    check_close_hinge_success,
    check_move_obstacle_success,
)
from tasks.preconditions import is_hinge_lid_already_open
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the put in hinge task.

    This task involves:
    1. Moving any fruit on top of the lid (obst_level=2)
    2. Opening the hinge box lid if closed (obst_level>=1)
    3. Picking up fruits and placing them into the hinge box
    4. Closing the lid

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing object_names, obst_object, and scene_objects.
        provider: PolicyProvider instance for subtask generation/execution.
        subtask_results: List to collect SubtaskResult.
        max_repairs: Max repair attempts per subtask.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        result: Task result ("full_success", "partial_success", or None).
    """
    if subtask_results is None:
        subtask_results = []

    print(f"[PutInHinge] Starting task with objects: {task.object_names}")
    print(f"[PutInHinge] Obstacle object on lid: {task.obst_object}")

    # Step 1: Move obstacle fruit off the lid (obst_level=2)
    if task.obst_object:
        print(f"\n[PutInHinge] Moving {task.obst_object} off the lid")

        try:
            # Pick the obstacle object
            result = provider.execute_subtask(
                subtask_name="pick_single",
                env=env,
                obj_name=task.obst_object,
                check_func=lambda e, o, t: check_pick_success(e, o),
                max_repairs=max_repairs,
            )
            subtask_results.append(result)

            # Place it aside (to a temporary position)
            hinge_body_pos = env.get_obj_pos("hinge_body")
            temp_pos = hinge_body_pos.copy()
            temp_pos[0] -= 0.15  # Move to the front
            temp_pos[2] += 0.05

            result = provider.execute_subtask(
                subtask_name="place_at_position",
                env=env,
                obj_name=task.obst_object,
                check_func=lambda e, o, t: check_move_obstacle_success(e, o),
                max_repairs=max_repairs,
                position=temp_pos,
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print(f"[PutInHinge] Skipping obstacle move for {task.obst_object}")
            if skip.result is not None:
                subtask_results.append(skip.result)

    # Step 2: Open the lid if closed (obst_level>=1)
    if is_hinge_lid_already_open(env) is False:
        print("[PutInHinge] Opening the hinge lid")
        try:
            result = provider.execute_subtask(
                subtask_name="open_hinge_lid",
                env=env,
                check_func=lambda e, o, t: check_open_hinge_success(e),
                max_repairs=max_repairs,
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print("[PutInHinge] Skipping open hinge lid")
            if skip.result is not None:
                subtask_results.append(skip.result)

    # Step 3: Pick and place each fruit into the hinge box
    for obj_name in task.object_names:
        print(f"\n[PutInHinge] Processing: {obj_name}")

        # Pick the fruit
        try:
            result = provider.execute_subtask(
                subtask_name="pick_single",
                env=env,
                obj_name=obj_name,
                check_func=lambda e, o, t: check_pick_success(e, o),
                max_repairs=max_repairs,
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print(f"[PutInHinge] Skipping pick for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue

        # Place into hinge box body
        try:
            result = provider.execute_subtask(
                subtask_name="place_single",
                env=env,
                obj_name=obj_name,
                target_name="hinge_body",
                check_func=lambda e, o, t: check_place_success(e, o, t),
                max_repairs=max_repairs,
                height_offset=0.15,
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print(f"[PutInHinge] Skipping place for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue

    # Step 4: Close the lid
    print("\n[PutInHinge] Closing the hinge lid")
    try:
        result = provider.execute_subtask(
            subtask_name="close_hinge_lid",
            env=env,
            check_func=lambda e, o, t: check_close_hinge_success(e),
            max_repairs=max_repairs,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[PutInHinge] Skipping close hinge lid")
        if skip.result is not None:
            subtask_results.append(skip.result)

    # Check result
    result = env.check_result()
    print(f"\n[PutInHinge] Task completed with result: {result}")
    return result
