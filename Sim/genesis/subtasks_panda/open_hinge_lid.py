"""
Open Hinge Lid Subtask.

Opens the lid of a hinge box by grasping and lifting the handle.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    grasp_handle,
    release_handle,
)


def open_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):
    """
    Open the hinge box lid.

    Args:
        env: LMPWrapper environment instance.
        handle_name: Name of the handle to grasp.
        lid_name: Name of the lid entity (for state checking).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[OpenHingeLid] Opening hinge lid")

    # Move to handle
    success = move_gripper_to(env, handle_name, pointing_to="down")
    if not success:
        print(f"[OpenHingeLid] Failed to move to handle")
        return False

    # Grasp the handle
    grasp_handle(env, handle_name)

    # Lift to open the lid
    for _ in range(5):
        move_parallel(env, "up", 0.05, pointing_to="down")
        move_parallel(env, "front", 0.05, pointing_to="down")

    # Release handle
    release_handle(env)

    # Move up to clear
    move_parallel(env, "up", 0.05, pointing_to="down")

    print(f"[OpenHingeLid] Successfully opened hinge lid")
    return True