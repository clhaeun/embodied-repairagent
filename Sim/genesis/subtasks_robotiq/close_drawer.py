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

def close_drawer(env, drawer_type="bottom"):
    drawer_handle_pos = env.get_obj_pos("top_drawer_handle") if drawer_type == "top" else env.get_obj_pos("bottom_drawer_handle")
    drawer_pos = env.get_obj_pos("drawer")
    drawer_bbox = env.get_obj_bbox("drawer")

    move_gripper_to(env, "top_drawer_handle", pointing_to="right")

    grasp_handle_robotiq85(env, "top_drawer_handle")

    move_parallel(env, "right", 0.12, pointing_to="right")

    release_handle_robotiq85(env)

    move_parallel(env, "left", 0.1, pointing_to="right")

    return True
