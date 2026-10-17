"""
Relocate Single Object Subtask.

Moves a misplaced object back to its correct location (e.g., chess piece to board).
"""

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_gripper,
    close_gripper,
)


def relocate_single(env, obj_name, target_name, pointing_to="down", height_offset=0.08,
                    target_position=None):
    """
    Relocate a misplaced object to its correct location.

    This subtask is used for objects that are in the wrong place
    (e.g., chess piece in wrong box) and need to be moved to the correct location.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to relocate.
        target_name: Name of the target location (e.g., "chess_board").
        pointing_to: Approach direction ("down", "left", "right").
        height_offset: Height above target to release.
        target_position: Optional specific position to place the object.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[RelocateSingle] Relocating {obj_name} to {target_name}")


    if not env.is_obj_visible(obj_name):
        print(f"[RelocateSingle] Object {obj_name} not visible")
        return False


    success = move_gripper_to(env, obj_name, pointing_to=pointing_to)
    if not success:
        print(f"[RelocateSingle] Failed to move to {obj_name}")
        return False

    close_gripper(env)


    move_parallel(env, "up", 0.1, pointing_to=pointing_to)


    if target_position is not None:
        place_pos = target_position.copy()
    else:

        target_bbox = env.get_obj_bbox(target_name)
        if target_bbox is None:
            print(f"[RelocateSingle] Target {target_name} not found")
            open_gripper(env)
            return False


        place_pos = (target_bbox[0] + target_bbox[1]) / 2
        place_pos[2] = target_bbox[1][2] + height_offset


    move_to_position(env, place_pos, pointing_to=pointing_to)


    open_gripper(env)


    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[RelocateSingle] Successfully relocated {obj_name} to {target_name}")
    return True