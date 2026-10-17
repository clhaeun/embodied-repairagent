"""
Release Container Subtask.

Releases a container held by gripper back to a position.
"""

from skill_code import (
    move_to_position,
    move_parallel,
    open_gripper,
)


def release_container(env, place_pos=None, original_pos=None, pointing_to="down", **kwargs):
    """
    Release a container held by gripper.

    Args:
        env: LMPWrapper environment instance.
        place_pos: Position to place the container [x, y, z].
        original_pos: Alternative name for place_pos (for compatibility).
        pointing_to: Approach direction.
        **kwargs: Additional arguments (ignored).

    Returns:
        True if successful, False otherwise.
    """

    if place_pos is None:
        place_pos = original_pos
    if place_pos is None:
        raise ValueError("Either place_pos or original_pos must be provided")

    print(f"[ReleaseContainer] Releasing at {place_pos}")


    release_pos = place_pos.copy()
    release_pos[2] += 0.15
    move_to_position(env, release_pos, pointing_to=pointing_to)


    release_pos[2] = place_pos[2] + 0.08
    move_to_position(env, release_pos, pointing_to=pointing_to)


    open_gripper(env)


    move_parallel(env, "up", 0.1, pointing_to=pointing_to)

    print(f"[ReleaseContainer] Successfully released")
    return True