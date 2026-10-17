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

def open_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):
    handle_pos = env.get_obj_pos(handle_name)
    lid_pos = env.get_obj_pos(lid_name)

    move_gripper_to(env, handle_name, pointing_to="down")

    grasp_handle_robotiq85(env, handle_name)

    for _ in range(5):
        move_parallel(env, "up", 0.05, pointing_to="down")
        move_parallel(env, "front", 0.05, pointing_to="down")

    release_handle_robotiq85(env)

    move_parallel(env, "up", 0.05, pointing_to="down")

    return True
