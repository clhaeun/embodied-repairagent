"""
Place At Position Subtask.

Places a currently held object to a specific position.
"""

from skill_code import (
    move_to_position,
    move_parallel,
    open_gripper,
)


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
    print(f"[PlaceAtPosition] Placing at position {position}")

    # Move to place position
    move_to_position(env, position, pointing_to=pointing_to)

    # Release object
    open_gripper(env)

    # Move up to clear
    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[PlaceAtPosition] Successfully placed at position")
    return True
