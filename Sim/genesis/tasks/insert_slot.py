"""
Insert Slot Task Execution.

Task: Remove the blocking object from target slot and insert the target object into it.

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.insert_slot import (
    check_remove_from_slot_success,
    check_place_beside_slot_success,
    check_insert_into_slot_success,
)
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the insert slot task.

    This task involves:
    1. Picking up the blocking object from the target slot
    2. Moving it beside the auxiliary slot
    3. Picking up the target object from the auxiliary slot
    4. Inserting it into the target slot

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

    print("[InsertSlot] Starting task")

    # Get positions
    target_slot_pos = env.get_obj_pos("target_slot")
    auxiliary_slot_pos = env.get_obj_pos("auxiliary_slot")

    print(f"[InsertSlot] Target slot position: {target_slot_pos}")
    print(f"[InsertSlot] Auxiliary slot position: {auxiliary_slot_pos}")

    # Step 1: Remove blocking object from target slot
    print("\n[InsertSlot] Step 1: Removing blocking object")
    try:
        result = provider.execute_subtask(
            subtask_name="remove_from_slot",
            env=env,
            obj_name="blocking_object",
            check_func=lambda e, o, t: check_remove_from_slot_success(e, o),
            max_repairs=max_repairs,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[InsertSlot] Skipping remove_from_slot")
        if skip.result is not None:
            subtask_results.append(skip.result)

    # Step 2: Place blocking object beside auxiliary slot
    print("\n[InsertSlot] Step 2: Placing blocking object aside")
    try:
        result = provider.execute_subtask(
            subtask_name="place_beside_slot",
            env=env,
            obj_name="blocking_object",
            target_name="auxiliary_slot",
            check_func=lambda e, o, t: check_place_beside_slot_success(e, o, t),
            max_repairs=max_repairs,
            slot_pos=auxiliary_slot_pos,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[InsertSlot] Skipping place_beside_slot")
        if skip.result is not None:
            subtask_results.append(skip.result)

    # Step 3: Insert target object into target slot
    print("\n[InsertSlot] Step 3: Inserting target object")
    try:
        result = provider.execute_subtask(
            subtask_name="insert_into_slot",
            env=env,
            obj_name="target_object",
            target_name="target_slot",
            check_func=lambda e, o, t: check_insert_into_slot_success(e, o, t),
            max_repairs=max_repairs,
            slot_pos=target_slot_pos,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print("[InsertSlot] Skipping insert_into_slot")
        if skip.result is not None:
            subtask_results.append(skip.result)

    # Check result
    result = env.check_result()
    print(f"\n[InsertSlot] Task completed with result: {result}")
    return result
