"""
Pour Liquid Task Execution.

Task: Pour the liquid from the beaker into the mug without spilling.

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.pour_liquid import (
    check_grasp_container_success,
    check_pour_success,
    check_release_container_success,
)
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the pour liquid task.

    This task involves:
    1. Grasping the beaker with vacuum gripper
    2. Moving it above the mug
    3. Tilting to pour the liquid
    4. Returning the beaker to original position

    Note: This task requires suction gripper (vacuum) to grasp the beaker.

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing scene_objects.
        provider: PolicyProvider instance for subtask generation/execution.
        subtask_results: List to collect SubtaskResult.
        max_repairs: Max repair attempts per subtask.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        result: Task result ("full_success" or None).
    """
    if subtask_results is None:
        subtask_results = []

    print("[PourLiquid] Starting task")


    beaker_pos = env.get_obj_pos("beaker")
    mug_pos = env.get_obj_pos("mug")

    print(f"[PourLiquid] Beaker position: {beaker_pos}")
    print(f"[PourLiquid] Mug position: {mug_pos}")


    print("\n[PourLiquid] Step 1: Grasping beaker")
    try:
        result = provider.execute_subtask(
            subtask_name="grasp_container_vacuum",
            env=env,
            obj_name="beaker",
            check_func=lambda e, o, t: check_grasp_container_success(e, o),
            max_repairs=max_repairs,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[PourLiquid] Skipping grasp_container_vacuum")
        if skip.result is not None:
            subtask_results.append(skip.result)


    print("\n[PourLiquid] Step 2: Pouring into mug")
    try:
        result = provider.execute_subtask(
            subtask_name="pour_into_container",
            env=env,
            target_name="mug",
            check_func=lambda e, o, t: check_pour_success(e, task),
            max_repairs=max_repairs,
            target_pos=mug_pos,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[PourLiquid] Skipping pour_into_container")
        if skip.result is not None:
            subtask_results.append(skip.result)


    print("\n[PourLiquid] Step 3: Releasing beaker")
    try:
        result = provider.execute_subtask(
            subtask_name="release_container_vacuum",
            env=env,
            obj_name="beaker",
            check_func=lambda e, o, t: check_release_container_success(e, o),
            max_repairs=max_repairs,
            original_pos=beaker_pos,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[PourLiquid] Skipping release_container_vacuum")
        if skip.result is not None:
            subtask_results.append(skip.result)


    result = env.check_result()
    print(f"\n[PourLiquid] Task completed with result: {result}")
    return result
