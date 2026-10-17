"""
Close Drawer Subtask.

Closes a prismatic drawer by grasping and pushing the handle.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    grasp_handle,
    release_handle,
)


def close_drawer(env, drawer_type="bottom"):
    """
    Close a prismatic drawer.

    Args:
        env: LMPWrapper environment instance.
        drawer_type: "top" or "bottom" drawer to close.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[CloseDrawer] Closing {drawer_type} drawer")

    # Get handle name based on drawer type
    handle_name = "top_drawer_handle" if drawer_type == "top" else "bottom_drawer_handle"

    # Move to handle (approach from right side)
    success = move_gripper_to(env, handle_name, pointing_to="right")
    if not success:
        print(f"[CloseDrawer] Failed to move to handle")
        return False

    # Grasp the handle
    grasp_handle(env, handle_name)

    # Push to close (right direction)
    move_parallel(env, "right", 0.12, pointing_to="right")

    # Release handle
    release_handle(env)

    # Move up to clear
    move_parallel(env, "left", 0.1, pointing_to="right")

    print(f"[CloseDrawer] Successfully closed {drawer_type} drawer")
    return True