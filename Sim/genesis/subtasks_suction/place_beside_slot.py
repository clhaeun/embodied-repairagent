import numpy as np

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    activate_vacuum,
    deactivate_vacuum,
    attach_vacuum_handle,
    detach_vacuum_handle,
    rotate_gripper,
)

def place_beside_slot(env, obj_name=None, target_name=None, slot_pos=None, offset_x=-0.08, **kwargs):
    obj_pos = env.get_obj_pos(obj_name)
    target_pos = env.get_obj_pos(target_name)
    slot_pos = env.get_obj_pos(target_name)

    aside_pos = slot_pos.copy()
    aside_pos[0] += offset_x
    aside_pos[2] += 0.1

    move_to_position(env, aside_pos, pointing_to="down")

    aside_pos[2] = slot_pos[2] + 0.02
    move_to_position(env, aside_pos, pointing_to="down")
    deactivate_vacuum(env)

    move_parallel(env, "up", 0.08, pointing_to="down")

    print(f"[PlaceBesideSlot] Successfully placed {obj_name}")
    return True
