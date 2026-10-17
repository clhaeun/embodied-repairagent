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

def place_beside_slot(env, obj_name=None, target_name=None, slot_pos=None, offset_x=-0.08, **kwargs):
    slot_pos = env.get_obj_pos(target_name)
    obj_pos = env.get_obj_pos(obj_name)

    aside_pos = slot_pos.copy()
    aside_pos[0] += offset_x
    aside_pos[2] += 0.1

    move_gripper_to(env, obj_name, pointing_to="down")
    close_robotiq85(env)
    move_parallel(env, "up", 0.15, pointing_to="down")
    aside_pos[2] = slot_pos[2] + 0.02
    move_to_position(env, aside_pos, pointing_to="down")
    open_robotiq85(env)
    move_parallel(env, "up", 0.08, pointing_to="down")

    print(f"[PlaceBesideSlot] Successfully placed {obj_name}")
    return True
