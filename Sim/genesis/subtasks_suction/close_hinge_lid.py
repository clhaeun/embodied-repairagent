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

def close_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):
    handle_pos = env.get_obj_pos(handle_name)
    lid_pos = env.get_obj_pos(lid_name)

    move_gripper_to(env, handle_name, pointing_to="down")
    activate_vacuum(env)
    attach_vacuum_handle(env, handle_name)

    for _ in range(3):
        move_parallel(env, "back", 0.05, pointing_to="down")
        move_parallel(env, "down", 0.05, pointing_to="down")

    detach_vacuum_handle(env)
    deactivate_vacuum(env)

    move_parallel(env, "up", 0.1, pointing_to="down")

    return True
