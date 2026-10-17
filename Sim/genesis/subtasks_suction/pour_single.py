"""
Pour Single Subtask (Suction Gripper).

Pours liquid from a source container (beaker) to a target container (mug).
Uses vacuum gripper to grasp the beaker.
"""

from skill_code import (
    move_to_position,
    move_parallel,
    rotate_gripper,
    activate_vacuum,
    deactivate_vacuum,
)


def pour_single(env, source_name="beaker", target_name="mug", pour_angle=120):
    """
    Pour liquid from source container to target container.

    This subtask uses vacuum gripper to grasp the source container,
    move it above the target, tilt to pour, then return it.

    Args:
        env: LMPWrapper environment instance.
        source_name: Name of the source container (default "beaker").
        target_name: Name of the target container (default "mug").
        pour_angle: Angle to tilt for pouring in degrees (default 120).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PourSingle] Pouring from {source_name} to {target_name}")


    source_pos = env.get_obj_pos(source_name)
    target_pos = env.get_obj_pos(target_name)

    if source_pos is None:
        print(f"[PourSingle] Source {source_name} not found")
        return False
    if target_pos is None:
        print(f"[PourSingle] Target {target_name} not found")
        return False

    print(f"[PourSingle] Source position: {source_pos}")
    print(f"[PourSingle] Target position: {target_pos}")


    approach_pos = source_pos.copy()
    approach_pos[2] += 0.15
    move_to_position(env, approach_pos, pointing_to="down")


    grasp_pos = source_pos.copy()
    grasp_pos[2] += 0.08
    move_to_position(env, grasp_pos, pointing_to="down")


    activate_vacuum(env)


    move_parallel(env, "up", 0.12, pointing_to="down")


    pour_pos = target_pos.copy()
    pour_pos[2] += 0.20
    pour_pos[1] -= 0.05
    move_to_position(env, pour_pos, pointing_to="down")


    rotate_gripper(env, angle=pour_angle, steps=60)


    for _ in range(100):
        env.env.step()


    rotate_gripper(env, angle=-pour_angle, steps=60)


    return_pos = source_pos.copy()
    return_pos[2] += 0.15
    move_to_position(env, return_pos, pointing_to="down")


    return_pos[2] = source_pos[2] + 0.08
    move_to_position(env, return_pos, pointing_to="down")


    deactivate_vacuum(env)


    move_parallel(env, "up", 0.1, pointing_to="down")

    print(f"[PourSingle] Successfully poured from {source_name} to {target_name}")
    return True
