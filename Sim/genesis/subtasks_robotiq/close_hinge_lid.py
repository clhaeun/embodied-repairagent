import numpy as np

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_robotiq85,
    close_robotiq85,
    pick_robotiq85,
    place_robotiq85,
    rotate_gripper,
    grasp_handle_robotiq85,
    release_handle_robotiq85,
)

def close_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):
    handle_pos = env.get_obj_pos(handle_name)
    lid_pos = env.get_obj_pos(lid_name)

    move_gripper_to(env, handle_name, pointing_to="down")
    close_robotiq85(env)

    move_gripper_to(env, handle_name, pointing_to="down")
    move_parallel(env, "back", 0.02, pointing_to="down")
    close_robotiq85(env)

    for _ in range(3):
        move_parallel(env, "back", 0.05, pointing_to="down")
        move_parallel(env, "down", 0.05, pointing_to="down")

    release_handle_robotiq85(env)

    move_parallel(env, "up", 0.1, pointing_to="down")

    return True
