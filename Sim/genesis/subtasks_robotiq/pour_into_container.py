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

def pour_into_container(env, target_pos, pour_angle=120, pour_steps=60, hold_steps=100):
    target_pos = env.get_obj_pos(target_name)
    handle_pos = env.get_obj_pos(handle_name)

    pour_pos = target_pos.copy()
    pour_pos[2] += 0.20
    pour_pos[1] -= 0.05  # Offset for pouring
    move_to_position(env, pour_pos, pointing_to="down")

    print(f"[PourIntoContainer] Tilting to pour")
    rotate_gripper(env, angle=pour_angle, steps=pour_steps)

    for _ in range(hold_steps):
        env.env.step()

    print(f"[PourIntoContainer] Returning upright")
    rotate_gripper(env, angle=-pour_angle, steps=pour_steps)

    print(f"[PourIntoContainer] Successfully poured")
    return True
