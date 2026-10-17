"""
Insert Into Slot Subtask.

Picks up an object and inserts it into a slot.
Used by insert_slot task.
"""

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_gripper,
    close_gripper,
)


def insert_into_slot(env, obj_name, slot_pos=None, target_name=None, insert_height=0.05, **kwargs):
    """
    Pick up an object and insert it into a slot.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to insert.
        slot_pos: Position of the target slot [x, y, z].
        target_name: Name of the target slot (optional, used if slot_pos is None).
        insert_height: Height above slot to release.
        **kwargs: Additional arguments (ignored).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[InsertIntoSlot] Inserting {obj_name}")

    if slot_pos is None:
        slot_pos = env.get_obj_pos(target_name)

    # Pick up the object
    move_gripper_to(env, obj_name, pointing_to="down")
    close_gripper(env)

    # Lift up
    move_parallel(env, "up", 0.15, pointing_to="down")

    # Move above slot
    insert_pos = slot_pos.copy()
    insert_pos[2] += 0.15
    move_to_position(env, insert_pos, pointing_to="down")

    # Lower into slot
    insert_pos[2] = slot_pos[2] + insert_height
    move_to_position(env, insert_pos, pointing_to="down")

    # Release
    open_gripper(env)

    # Move up to clear
    move_parallel(env, "up", 0.1, pointing_to="down")

    print(f"[InsertIntoSlot] Successfully inserted {obj_name}")
    return True