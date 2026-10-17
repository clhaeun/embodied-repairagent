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

def pour_into_container(env, target_pos, pour_angle=120, pour_steps=60, hold_steps=100):
    target_pos = env.get_obj_pos(target_name)
    handle_pos = env.get_obj_pos(handle_name)

    pour_pos = target_pos.copy()
    pour_pos[2] += 0.20
    pour_pos[1] -= 0.05
    move_to_position(env, pour_pos, pointing_to="down")

    print(f"[PourIntoContainer] Tilting to pour")
    rotate_gripper(env, angle=pour_angle, steps=pour_steps)

    for _ in range(hold_steps):
        env.env.step()

    print(f"[PourIntoContainer] Returning upright")
    rotate_gripper(env, angle=-pour_angle, steps=pour_steps)

    print(f"[PourIntoContainer] Successfully poured")
    return True
