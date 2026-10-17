import numpy as np

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    activate_vacuum,
    deactivate_vacuum,
    attach_vacuum_handle,
    detach_vacuum_handle,
    rotate_gripper,
)

def close_drawer(env, drawer_type="bottom"):
    drawer_handle_pos = env.get_obj_pos("top_drawer_handle")
    drawer_handle_pos[2] += 0.05
    drawer_pos = env.get_obj_pos("drawer")
    drawer_pos[2] += 0.05
    drawer_pos[1] += 0.05
    drawer_pos[0] += 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.05
    drawer_pos[0] -= 0.05
    drawer_pos[2] -= 0.05
    drawer_pos[1] -= 0.
