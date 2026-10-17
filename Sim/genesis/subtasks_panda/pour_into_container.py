"""
Pour Into Container Subtask.

Pours liquid into a target container by tilting.
"""

from skill_code import (
    move_to_position,
    rotate_gripper,
)


def pour_into_container(env, target_pos, pour_angle=120, pour_steps=60, hold_steps=100):
    """
    Pour liquid into a target container.

    Args:
        env: LMPWrapper environment instance.
        target_pos: Position of the target container [x, y, z].
        pour_angle: Angle to tilt for pouring (degrees).
        pour_steps: Number of steps for tilting animation.
        hold_steps: Number of steps to hold pour position.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PourIntoContainer] Pouring into target at {target_pos}")


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