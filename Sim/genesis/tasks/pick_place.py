"""
Pick and Place Task Execution.

Task: Place cube blocks into the toy box.
Handles stacked blocks by unstacking them first.

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.pick_place import (
    check_pick_success,
    check_place_success,
    check_unstack_success,
)
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the pick and place task.

    This task involves picking up cube blocks and placing them into a toy box.
    If blocks are stacked (obst_level > 0), unstacks them first in reverse order.

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing object_names, stack_pairs, and scene_objects.
        provider: PolicyProvider instance for subtask generation/execution.
        subtask_results: List to collect SubtaskResult.
        max_repairs: Max repair attempts per subtask.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        result: Task result ("full_success", "partial_success", or None).
    """
    if subtask_results is None:
        subtask_results = []

    print(f"[PickPlace] Starting task with objects: {task.object_names}")
    print(f"[PickPlace] Stack pairs: {task.stack_pairs}")
    
    # Step 1: Unstack blocks if any (process in reverse order - top first)
    if task.stack_pairs:
        print(f"\n[PickPlace] Unstacking {len(task.stack_pairs)} stacked block(s)")
        for low_obj, top_obj in reversed(task.stack_pairs):
            print(f"[PickPlace] Unstacking {top_obj} from {low_obj}")
            try:
                result = provider.execute_subtask(
                    subtask_name="unstack_single",
                    env=env,
                    obj_name=top_obj,
                    check_func=lambda e, o, t: check_unstack_success(e, o),
                    max_repairs=max_repairs,
                )
                subtask_results.append(result)
            except SubtaskSkip as skip:
                print(f"[PickPlace] Skipping unstack for {top_obj}")
                if skip.result is not None:
                    subtask_results.append(skip.result)
                continue

    print("\n[PickPlace] Starting pick and place of blocks into toy box")
    # Step 2: Pick and place the base block into the toy box
    base_block = task.object_names[0]
    print(f"\n[PickPlace] Picking base block: {base_block}")
    try:
        result = provider.execute_subtask(
            subtask_name="pick_single",
            env=env,
            obj_name=base_block,
            check_func=lambda e, o, t: check_pick_success(e, o),
            max_repairs=max_repairs,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print(f"[PickPlace] Skipping pick for {base_block}")
        if skip.result is not None:
            subtask_results.append(skip.result)

    print(f"[PickPlace] Placing base block {base_block} into toy box")
    try:
        result = provider.execute_subtask(
            subtask_name="place_single",
            env=env,
            obj_name=base_block,
            target_name="toy_box",
            check_func=lambda e, o, t: check_place_success(e, o, t),
            max_repairs=max_repairs,
            height_offset=0.1,
        )
        subtask_results.append(result)
    except SubtaskSkip as skip:
        print(f"[PickPlace] Skipping place for {base_block}")
        if skip.result is not None:
            subtask_results.append(skip.result)

    remain_blocks = task.object_names[1:]
    # Step 3: Pick and place each block into the toy box
    for obj_name in remain_blocks:
        print(f"\n[PickPlace] Processing: {obj_name}")

        # Pick the block
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
            print(f"[PickPlace] Skipping pick for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue

        # Place into toy box
        try:
            result = provider.execute_subtask(
                subtask_name="place_single",
                env=env,
                obj_name=obj_name,
                target_name="toy_box",
                check_func=lambda e, o, t: check_place_success(e, o, t),
                max_repairs=max_repairs,
                height_offset=0.1,
            )
            subtask_results.append(result)
        except SubtaskSkip as skip:
            print(f"[PickPlace] Skipping place for {obj_name}")
            if skip.result is not None:
                subtask_results.append(skip.result)
            continue

    # Check result
    result = env.check_result()
    print(f"\n[PickPlace] Task completed with result: {result}")
    return result
