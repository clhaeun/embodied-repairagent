"""
Grasp Container Subtask.

Grasps a container using gripper for pouring tasks.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    close_gripper,
)


def grasp_container(env, obj_name, pointing_to="down"):
    """
    Grasp a container using gripper.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the container to grasp.
        pointing_to: Approach direction.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[GraspContainer] Grasping {obj_name}")


    success = move_gripper_to(env, obj_name, pointing_to=pointing_to)
    if not success:
        print(f"[GraspContainer] Failed to move to {obj_name}")
        return False


    close_gripper(env)


    move_parallel(env, "up", 0.12, pointing_to=pointing_to)

    print(f"[GraspContainer] Successfully grasped {obj_name}")
    return True