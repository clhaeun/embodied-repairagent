"""
Remove From Slot Subtask.

Picks up an object from a slot and lifts it.
Used by insert_slot task.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    close_gripper,
)


def remove_from_slot(env, obj_name, lift_height=0.15, **kwargs):
    """
    Pick up an object from a slot and lift it.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to remove.
        lift_height: Height to lift after picking.
        **kwargs: Additional arguments (ignored).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[RemoveFromSlot] Removing {obj_name}")

    # Move to object and pick
    move_gripper_to(env, obj_name, pointing_to="down")
    close_gripper(env)

    # Lift up
    move_parallel(env, "up", lift_height, pointing_to="down")

    print(f"[RemoveFromSlot] Successfully removed {obj_name}")
    return True