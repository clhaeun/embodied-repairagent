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

def release_container_vacuum(env, original_pos, pointing_to="down"):
    gripper_pos = env.get_obj_pos("gripper")
    target_pos = env.get_obj_pos("target")
    handle_pos = env.get_obj_pos("handle")

    move_gripper_to(env, "target", pointing_to="down")
    close_robotiq85(env)

    move_parallel(env, "up", 0.15, pointing_to="down")
    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")
    open_robotiq85(env)
    move_parallel(env, "up", 0.08, pointing_to="down")

    move_gripper_to(env, "handle", pointing_to="right")
    grasp_handle_robotiq85(env, "handle")

    move_parallel(env, "left", 0.12, pointing_to="right")
    release_handle_robotiq85(env)
