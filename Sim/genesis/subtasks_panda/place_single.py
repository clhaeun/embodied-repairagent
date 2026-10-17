"""
Place Single Object Subtask.

Places a currently held object to a target location.
"""

from skill_code import (
    move_to_position,
    move_parallel,
    open_gripper,
)


def place_single(env, target_name, pointing_to="down", height_offset=0.1):
    """
    Place a held object at a target location.

    Args:
        env: LMPWrapper environment instance.
        target_name: Name of the target object/container.
        pointing_to: Approach direction ("down", "left", "right").
        height_offset: Height above target to release (default 0.1m).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PlaceSingle] Placing at {target_name}")


    target_pos = env.get_obj_pos(target_name)


    place_pos = target_pos.copy()
    place_pos[2] += height_offset


    move_to_position(env, place_pos, pointing_to=pointing_to)


    open_gripper(env)


    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[PlaceSingle] Successfully placed at {target_name}")
    return True


def place_at_position(env, position, pointing_to="down"):
    """
    Place a held object at a specific position.

    Args:
        env: LMPWrapper environment instance.
        position: Target position [x, y, z].
        pointing_to: Approach direction ("down", "left", "right").

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PlaceSingle] Placing at position {position}")


    move_to_position(env, position, pointing_to=pointing_to)


    open_gripper(env)


    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[PlaceSingle] Successfully placed at position")
    return True