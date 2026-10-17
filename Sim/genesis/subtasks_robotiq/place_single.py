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

def place_single(env, target_name, pointing_to="down", height_offset=0.1):
    target_pos = env.get_obj_pos(target_name)
    obj_pos = env.get_obj_pos(obj_name)

    move_gripper_to(env, obj_name, pointing_to="down")
    close_robotiq85(env)

    move_parallel(env, "up", 0.15, pointing_to="down")

    target_pos[2] += height_offset
    move_to_position(env, target_pos, pointing_to="down")

    open_robotiq85(env)

    move_parallel(env, "up", 0.08, pointing_to="down")

    print(f"[PlaceSingle] Successfully placed at {target_name}")
    return True
