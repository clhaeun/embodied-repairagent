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

def unstack_single(env, obj_name, pointing_to="down"):
    obj_pos = env.get_obj_pos(obj_name)
    handle_pos = env.get_obj_pos(handle_name)

    move_gripper_to(env, obj_name, pointing_to=pointing_to)
    activate_vacuum(env)
    attach_vacuum_handle(env, handle_name)

    move_parallel(env, "up", 0.1, pointing_to=pointing_to)

    aside_pos = obj_pos.copy()
    aside_pos[1] += 0.08  # Move to the right side
    aside_pos[2] = obj_pos[2] + 0.05  # Slightly above floor level

    move_to_position(env, aside_pos, pointing_to=pointing_to)

    detach_vacuum_handle(env)
    deactivate_vacuum(env)

    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[UnstackSingle] Successfully unstacked {obj_name}")
    return True
