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

def remove_from_slot(env, obj_name, lift_height=0.15, **kwargs):
    obj_pos = env.get_obj_pos(obj_name)
    handle_pos = env.get_obj_pos(handle_name)

    move_gripper_to(env, obj_name, pointing_to="down")
    activate_vacuum(env)
    attach_vacuum_handle(env, handle_name)

    move_parallel(env, "up", 0.15, pointing_to="down")

    target_pos = env.get_obj_pos(target_name)
    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")
    deactivate_vacuum(env)
    move_parallel(env, "up", 0.08, pointing_to="down")

    detach_vacuum_handle(env)
    deactivate_vacuum(env)

    return True
