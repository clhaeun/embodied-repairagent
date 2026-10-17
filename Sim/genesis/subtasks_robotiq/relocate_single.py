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

def relocate_single(env, obj_name, target_name, pointing_to="down", height_offset=0.08, target_position=None):
    obj_pos = env.get_obj_pos(obj_name)
    target_pos = env.get_obj_pos(target_name)

    move_gripper_to(env, obj_name, pointing_to=pointing_to)
    close_robotiq85(env)

    move_parallel(env, "up", 0.1, pointing_to=pointing_to)

    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to=pointing_to)

    open_robotiq85(env)

    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    return True
