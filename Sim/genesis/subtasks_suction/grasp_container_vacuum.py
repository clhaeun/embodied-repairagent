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

def grasp_container_vacuum(env, obj_name, pointing_to="down"):
    obj_pos = env.get_obj_pos(obj_name)
    target_pos = env.get_obj_pos(target_name)
    move_gripper_to(env, obj_name, pointing_to='down', depth=0.01)
    move_to_position(env, pos, pointing_to='down', lift_clearance=0.12)
    move_parallel(env, direction, offset, pointing_to='down')
    activate_vacuum(env)
    deactivate_vacuum(env)
    attach_vacuum_handle(env, handle_name)
    detach_vacuum_handle(env)
    rotate_gripper(env, angle, steps=60)
    move_gripper_to(env, obj_name, pointing_to='down')
    activate_vacuum(env)
    move_parallel(env, 'up', 0.15, pointing_to='down')
    move_to_position(env, target_pos, pointing_to='down')
    deactivate_vacuum(env)
    move_parallel(env, 'up', 0.08, pointing_to='down')
    move_gripper_to(env, box_handle, pointing_to='down')
    activate_vacuum(env)
    attach_vacuum_handle(env, box_handle)
    move_parallel(env, 'up', 0.08, pointing_to='down')
    move_parallel(env, 'front', 0.08, pointing_to='down')
    detach_vacuum_handle(env)
    deactivate_vacuum(env)
    move_gripper_to(env, handle_name, pointing_to='right')
    activate_vacuum(env)
    attach_vacuum_handle(env, handle_name)
    move_parallel(env, 'left', 0.12, pointing_to='right')
    detach_vacuum_handle(env)
    deactivate_vacuum(env)


    move_gripper_to(env, obj_name, pointing_to="down")

    activate_vacuum(env)

    move_parallel(env, "up", 0.15, pointing_to="down")
    target_pos[2] += 0.10
    move_to_position(env, target_pos, pointing_to="down")

    deactivate_vacuum(env)

    move_parallel(env, "up", 0.08, pointing_to="down")
