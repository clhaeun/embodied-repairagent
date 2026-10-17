"""
Put in Prismatic Drawer Task Execution.

Task: Place cylinder blocks into the prismatic drawer.
Handles obstacles: closes blocking top drawer, opens closed bottom drawer.

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.put_in_prismatic import (
    check_pick_success,
    check_place_success,
    check_open_drawer_success,
    check_close_drawer_success,
)
from tasks.preconditions import is_drawer_already_closed, is_drawer_already_open
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the put in prismatic drawer task.

    This task involves:
    1. Closing the top drawer if it's open and blocking (obst_level=2)
    2. Opening the bottom drawer if closed (obst_level>=1)
    3. Picking up cylinder blocks and placing them into the drawer
    4. Closing the bottom drawer

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing object_names and scene_objects.
        provider: PolicyProvider instance for subtask generation/execution.
        subtask_results: List to collect SubtaskResult.
        max_repairs: Max repair attempts per subtask.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        result: Task result ("full_success", "partial_success", or None).
    """
    if subtask_results is None:
        subtask_results = []

    print(f"[PutInPrismatic] Starting task with objects: {task.object_names}")

    # Step 1: Close top drawer if open (obst_level=2)
    if is_drawer_already_closed(env, "top") is False:
        print("[PutInPrismatic] Closing top drawer first")
        try:
            result = provider.execute_subtask(
                subtask_name="close_drawer",
                env=env,
                target_name="top",
                check_func=lambda e, o, t: check_close_drawer_success(e, "top"),
                max_repairs=max_repairs,
                drawer_type="top",
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print("[PutInPrismatic] Skipping close top drawer")
            if skip.result is not None:
                subtask_results.append(skip.result)

    # Step 2: Open bottom drawer if closed (obst_level>=1)
    if is_drawer_already_open(env, "bottom") is False:
        print("[PutInPrismatic] Opening bottom drawer")
        try:
            result = provider.execute_subtask(
                subtask_name="open_drawer",
                env=env,
                target_name="bottom",
                check_func=lambda e, o, t: check_open_drawer_success(e, "bottom"),
                max_repairs=max_repairs,
                drawer_type="bottom",
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print("[PutInPrismatic] Skipping open bottom drawer")
            if skip.result is not None:
                subtask_results.append(skip.result)

    # Step 3: Pick and place each cylinder into the drawer
    for obj_name in task.object_names:
        print(f"\n[PutInPrismatic] Processing: {obj_name}")

        # Pick the cylinder
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
            print(f"[PutInPrismatic] Skipping pick for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue

        # Place into bottom drawer
        try:
            result = provider.execute_subtask(
                subtask_name="place_single",
                env=env,
                obj_name=obj_name,
                target_name="bottom_drawer",
                check_func=lambda e, o, t: check_place_success(e, o, t),
                max_repairs=max_repairs,
                height_offset=0.1,
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print(f"[PutInPrismatic] Skipping place for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue

    # Step 4: Close the bottom drawer
    print("\n[PutInPrismatic] Closing bottom drawer")
    try:
        result = provider.execute_subtask(
            subtask_name="close_drawer",
            env=env,
            target_name="bottom",
            check_func=lambda e, o, t: check_close_drawer_success(e, "bottom"),
            max_repairs=max_repairs,
            drawer_type="bottom",
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[PutInPrismatic] Skipping close bottom drawer")
        if skip.result is not None:
            subtask_results.append(skip.result)

    # Check result
    result = env.check_result()
    print(f"\n[PutInPrismatic] Task completed with result: {result}")
    return result
