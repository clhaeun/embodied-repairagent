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

def remove_from_slot(env, obj_name, lift_height=0.15, **kwargs):
    obj_pos = env.get_obj_pos(obj_name)
    slot_pos = env.get_obj_pos(slot_name)
    print(f'[RemoveFromSlot] Removing {obj_name}')
    move_gripper_to(env, obj_name, pointing_to='down')
    close_gripper(env)
    move_parallel(env, 'up', lift_height, pointing_to='down')
    print(f'[RemoveFromSlot] Successfully removed {obj_name}')
    pass


    move_gripper_to(env, obj_name, pointing_to="down")
    close_robotiq85(env)

    move_parallel(env, "up", 0.15, pointing_to="down")

    target_pos = slot_pos.copy()
    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")
    open_robotiq85(env)
    move_parallel(env, "up", 0.08, pointing_to="down")

    return True
