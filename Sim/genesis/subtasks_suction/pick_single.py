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

def pick_single(env, obj_name, pointing_to="down"):
    obj_pos = env.get_obj_pos(obj_name)
    target_pos = env.get_obj_pos(target_name)

    move_gripper_to(env, obj_name, pointing_to=pointing_to)

    activate_vacuum(env)

    move_parallel(env, "up", 0.15, pointing_to="down")

    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")

    deactivate_vacuum(env)

    move_parallel(env, "up", 0.08, pointing_to="down")

    return True
