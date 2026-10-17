"""
Unstack Single Object Subtask.

Removes the top object from a stack and places it aside.
"""

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_gripper,
    close_gripper,
)


def unstack_single(env, obj_name, pointing_to="down"):
    """
    Remove the top object from a stack and place it aside.

    This subtask picks the object on top of a stack and moves it
    to a nearby clear location so the objects below become accessible.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the top object to unstack.
        pointing_to: Approach direction ("down", "left", "right").

    Returns:
        True if successful, False otherwise.
    """
    print(f"[UnstackSingle] Unstacking {obj_name}")


    if not env.is_obj_visible(obj_name):
        print(f"[UnstackSingle] Object {obj_name} not visible")
        return False


    top_pos = env.get_obj_pos(obj_name)


    success = move_gripper_to(env, obj_name, pointing_to=pointing_to)
    if not success:
        print(f"[UnstackSingle] Failed to move to {obj_name}")
        return False

    close_gripper(env)


    move_parallel(env, "up", 0.1, pointing_to=pointing_to)


    aside_pos = top_pos.copy()
    aside_pos[1] += 0.08
    aside_pos[2] = top_pos[2] + 0.05


    move_to_position(env, aside_pos, pointing_to=pointing_to)


    open_gripper(env)


    move_parallel(env, "up", 0.08, pointing_to=pointing_to)

    print(f"[UnstackSingle] Successfully unstacked {obj_name}")
    return True