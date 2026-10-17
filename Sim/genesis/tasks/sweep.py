"""
Sweep Chess Pieces Task Execution.

Task: Sweep chess pieces into their corresponding boxes
(white pieces to left box, black pieces to right box).
Handles misplaced pieces by relocating them to the board first.

Note: This task code uses provider.execute_subtask() for subtask execution.
      The provider is injected into the execution context by execute_task().
"""

from tasks.checks.sweep import (
    check_sweep_success,
    check_relocate_success,
)
from exceptions import SubtaskSkip


def run_task(env, task, provider=None, subtask_results=None, max_repairs=5, **kwargs):
    """
    Execute the sweep chess pieces task.

    This task involves sweeping chess pieces into their corresponding boxes:
    - White pieces go to the left box
    - Black pieces go to the right box

    If pieces are misplaced (obst_level > 0), relocates them to the board first.

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing object_names, piece_to_box, obst_pieces.
        provider: PolicyProvider instance for subtask generation/execution.
        subtask_results: List to collect SubtaskResult.
        max_repairs: Max repair attempts per subtask.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        result: Task result ("full_success", "partial_success", or None).
    """
    if subtask_results is None:
        subtask_results = []

    print(f"[Sweep] Starting task with pieces: {task.object_names}")
    print(f"[Sweep] Piece to box mapping: {task.piece_to_box}")
    print(f"[Sweep] Misplaced pieces: {task.obst_pieces}")

    # Step 1: Relocate misplaced pieces back to the board
    if task.obst_pieces:
        print(f"\n[Sweep] Relocating {len(task.obst_pieces)} misplaced piece(s)")

        # Calculate grid positions to avoid collisions
        board_bbox = env.get_obj_bbox("board")
        if board_bbox is not None:
            center = (board_bbox[0] + board_bbox[1]) / 2
            bbox_size = board_bbox[1] - board_bbox[0]
            spacing = min(bbox_size[0], bbox_size[1]) * 0.3
            height = board_bbox[1][2] + 0.08

            grid_offsets = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, 1), (1, -1), (-1, -1),
            ]

        for i, piece_name in enumerate(task.obst_pieces):
            print(f"[Sweep] Relocating {piece_name} back to board")

            # Calculate target position with offset
            target_pos = None
            if board_bbox is not None and i < len(grid_offsets):
                offset = grid_offsets[i]
                target_pos = center.copy()
                target_pos[0] += offset[0] * spacing
                target_pos[1] += offset[1] * spacing
                target_pos[2] = height
                print(f"[Sweep] Target position for {piece_name}: offset {offset}")

            try:
                result = provider.execute_subtask(
                    subtask_name="relocate_single",
                    env=env,
                    obj_name=piece_name,
                    target_name="board",
                    target_position=target_pos,
                    check_func=lambda e, o, t: check_relocate_success(e, o, t),
                    max_repairs=max_repairs,
                )
                subtask_results.append(result)
            except SubtaskSkip:
                print(f"[Sweep] Skipping relocate for {piece_name}")
                continue

    # Step 2: Sweep all pieces to their corresponding boxes
    for piece_name in task.object_names:
        print(f"\n[Sweep] Processing: {piece_name}")

        # Get target box from piece_to_box mapping
        target_box = task.piece_to_box.get(piece_name)
        if target_box is None:
            print(f"[Sweep] No target box for {piece_name}, skipping")
            continue

        # Sweep piece to target box
        try:
            result = provider.execute_subtask(
                subtask_name="sweep_single",
                env=env,
                obj_name=piece_name,
                target_name=target_box,
                check_func=lambda e, o, t: check_sweep_success(e, o, t),
                max_repairs=max_repairs,
            )
            subtask_results.append(result)
        except SubtaskSkip:
            print(f"[Sweep] Skipping sweep for {piece_name}")
            continue

    # Check result
    result = env.check_result()
    print(f"\n[Sweep] Task completed with result: {result}")
    return result