"""
Pick Single Object Subtask.

Picks up a single object using gripper primitives.
"""

from skill_code import (
    move_gripper_to,
    close_gripper,
)


def pick_single(env, obj_name, pointing_to="down"):
    """
    Pick up a single object.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to pick.
        pointing_to: Approach direction ("down", "left", "right").

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PickSingle] Picking {obj_name}")

    # Move gripper to object
    success = move_gripper_to(env, obj_name, pointing_to=pointing_to)
    if not success:
        print(f"[PickSingle] Failed to move to {obj_name}")
        return False

    # Close gripper to grasp
    close_gripper(env)

    print(f"[PickSingle] Successfully picked {obj_name}")
    return True