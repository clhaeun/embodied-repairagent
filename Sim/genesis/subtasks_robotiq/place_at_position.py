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

def place_at_position(env, position, pointing_to="down"):
    target_pos = env.get_obj_pos(target_name)
    handle_pos = env.get_obj_pos(handle_name)

    move_gripper_to(env, obj_name, pointing_to="down")
    close_robotiq85(env)
    move_parallel(env, "up", 0.15, pointing_to="down")
    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")
    open_robotiq85(env)
    move_parallel(env, "up", 0.08, pointing_to="down")

    move_gripper_to(env, handle_name, pointing_to="down")
    grasp_handle_robotiq85(env, handle_name)
    for _ in range(5):
        move_parallel(env, "up", 0.08, pointing_to="down")
        move_parallel(env, "front", 0.08, pointing_to="down")
    release_handle_robotiq85(env)

    move_gripper_to(env, handle_name, pointing_to="right")
    grasp_handle_robotiq85(env, handle_name)
    move_parallel(env, "left", 0.12, pointing_to="right")
    release_handle_robotiq85(env)

    move_gripper_to(env, obj_name, pointing_to="down")
    close_robotiq85(env)
    move_parallel(env, "up", 0.15, pointing_to="down")
    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")
    open_robotiq85(env)
    move_parallel(env, "up", 0.08, pointing_to="down")

    print(f"[PlaceAtPosition] Successfully placed at position")
    return True
