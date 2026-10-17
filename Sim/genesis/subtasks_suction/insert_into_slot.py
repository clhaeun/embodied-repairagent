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

def insert_into_slot(env, obj_name, slot_pos=None, target_name=None, insert_height=0.05, **kwargs):
    obj_pos = env.get_obj_pos(obj_name)
    slot_pos = env.get_obj_pos(target_name)

    move_gripper_to(env, obj_name, pointing_to="down")
    activate_vacuum(env)

    move_parallel(env, "up", 0.15, pointing_to="down")

    insert_pos = slot_pos.copy()
    insert_pos[2] += 0.15
    move_to_position(env, insert_pos, pointing_to="down")

    insert_pos[2] = slot_pos[2] + insert_height
    move_to_position(env, insert_pos, pointing_to="down")

    deactivate_vacuum(env)

    move_parallel(env, "up", 0.1, pointing_to="down")

    print(f"[InsertIntoSlot] Successfully inserted {obj_name}")
    return True
