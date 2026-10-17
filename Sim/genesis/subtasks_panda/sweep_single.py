"""
Sweep Single Object Subtask.

Moves a chess piece into a target box using pick-and-place.
"""

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_gripper,
    close_gripper,
)


def sweep_single(env, obj_name, target_name, pointing_to="down"):
    """
    Move a chess piece into a target box using pick-and-place.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the chess piece to move.
        target_name: Name of the target box ("left_chess_box" or "right_chess_box").
        pointing_to: Approach direction ("down", "left", "right").

    Returns:
        True if successful, False otherwise.
    """
    print(f"[SweepSingle] Sweeping {obj_name} to {target_name}")

    # Move to and grasp the piece
    success = move_gripper_to(env, obj_name, pointing_to=pointing_to)
    if not success:
        print(f"[SweepSingle] Failed to move to {obj_name}")
        return False

    close_gripper(env)

    # Lift up
    move_parallel(env, "up", 0.1, pointing_to=pointing_to)

    # Get target box center position
    target_bbox = env.get_obj_bbox(target_name)
    if target_bbox is None:
        print(f"[SweepSingle] Target {target_name} not found")
        open_gripper(env)
        return False

    # Calculate place position at center of target box
    place_pos = (target_bbox[0] + target_bbox[1]) / 2
    place_pos[2] = target_bbox[1][2] + 0.05  # slightly above box

    # Move to target
    move_to_position(env, place_pos, pointing_to=pointing_to)

    # Release piece
    open_gripper(env)

    # Move up to clear
    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[SweepSingle] Successfully swept {obj_name} to {target_name}")
    return True