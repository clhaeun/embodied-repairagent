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

def pick_and_place_object(env, obj_name, target_pos, place_height=0.01, spread_offset=0.0, placed_objects=None, use_smart_placement=True):
    obj_pos = env.get_obj_pos(obj_name)
    target_pos = env.get_obj_pos(target_name)

    open_robotiq85(env)

    depth = 0.03 if "ball" in obj_name else 0.01
    move_gripper_to(env, obj_name, pointing_to="down", depth=depth)
    close_robotiq85(env)

    move_parallel(env, "up", 0.20, pointing_to="down")

    if use_smart_placement and placed_objects:
        place_pos = find_place_position(env, target_pos, placed_objects)
        print(f"[PickAndPlace] Smart placement: found position {place_pos[:2]}")
    else:
        place_pos = target_pos.copy()
        place_pos[0] += spread_offset

    place_pos[2] = target_pos[2] + 0.12

    move_to_position(env, place_pos, pointing_to="down")

    place_pos[2] = target_pos[2] + place_height
    move_to_position(env, place_pos, pointing_to="down")

    open_robotiq85(env)

    move_parallel(env, "up", 0.08, pointing_to="down")

    print(f"[PickAndPlace] Successfully placed {obj_name}")
    return True
