"""
Place Beside Slot Subtask.

Places a held object beside a slot.
Used by insert_slot task.
"""

from skill_code import (
    move_to_position,
    move_parallel,
    open_gripper,
)


def place_beside_slot(env, obj_name=None, target_name=None, slot_pos=None, offset_x=-0.08, **kwargs):
    """
    Place a held object beside a slot.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object being placed (optional, for logging).
        target_name: Name of the target slot (optional).
        slot_pos: Position of the slot [x, y, z].
        offset_x: X-axis offset from slot.
        **kwargs: Additional arguments (ignored).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PlaceBesideSlot] Placing {obj_name} beside {target_name}")

    if slot_pos is None:
        if target_name:
            slot_pos = env.get_obj_pos(target_name)
        else:
            raise ValueError("Either slot_pos or target_name must be provided")


    aside_pos = slot_pos.copy()
    aside_pos[0] += offset_x
    aside_pos[2] += 0.1


    move_to_position(env, aside_pos, pointing_to="down")


    aside_pos[2] = slot_pos[2] + 0.02
    move_to_position(env, aside_pos, pointing_to="down")
    open_gripper(env)


    move_parallel(env, "up", 0.1, pointing_to="down")

    print(f"[PlaceBesideSlot] Successfully placed {obj_name}")
    return True