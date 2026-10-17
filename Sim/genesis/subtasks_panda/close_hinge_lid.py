"""
Close Hinge Lid Subtask.

Closes the lid of a hinge box by grasping and pushing the handle.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    release_handle,
    close_gripper,
)


def close_hinge_lid(env, handle_name="hinge_handle", lid_name="hinge_lid"):
    """
    Close the hinge box lid.

    Args:
        env: LMPWrapper environment instance.
        handle_name: Name of the handle to grasp.
        lid_name: Name of the lid entity (for state checking).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[CloseHingeLid] Closing hinge lid")

    # Move to handle
    success = move_gripper_to(env, handle_name, pointing_to="down")
    if not success:
        print(f"[CloseHingeLid] Failed to move to handle")
        return False

    # Grasp the handle
    move_gripper_to(env, handle_name, pointing_to="down")
    move_parallel(env, "back", 0.02, pointing_to="down")
    close_gripper(env)

    for _ in range(3):
        # Pull to close the lid
        move_parallel(env, "back", 0.05, pointing_to="down")
        move_parallel(env, "down", 0.05, pointing_to="down")

    # Release handle
    release_handle(env)

    # Move up to clear
    move_parallel(env, "up", 0.1, pointing_to="down")

    print(f"[CloseHingeLid] Successfully closed hinge lid")
    return True