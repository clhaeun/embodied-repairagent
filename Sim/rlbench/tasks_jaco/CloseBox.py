import numpy as np
from time import sleep
from pyrep.objects.shape import Shape
from scipy.spatial.transform import Rotation as R

from skill_code_jaco import (
    jaco_move_linear,
    jaco_align_gripper,
    jaco_close_gripper,
    jaco_rotate_gripper
)


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    Test Jaco alignment and rotation using CloseBox task.
    Align gripper properly to close the box lid.
    """
    sleep(1.0)

    lid = Shape('box_lid')
    lid_pos = np.array(lid.get_position(), dtype=float)

    print("[CloseBox] Aligning gripper for box lid...")
    obs, reward, done = jaco_align_gripper(
        env, task,
        approach_direction="right",
        maintain_position=True,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[CloseBox] Closing gripper...")
    obs, reward, done = jaco_close_gripper(
        env, task,
        maintain_pose=True,
        n_steps=8,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    approach_pos = lid_pos + np.array([0.15, 0.0, 0.0], dtype=float)
    print(f"[CloseBox] Moving to lid approach: {approach_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=approach_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print(f"[CloseBox] Moving to lid: {lid_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=lid_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    close_pos = lid_pos + np.array([-0.1, 0.0, 0.0], dtype=float)
    print(f"[CloseBox] Closing box lid: {close_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=close_pos,
        n_waypoints=10,
        timeout_s=10.0
    )

    if done:
        print("[CloseBox] Success! Box has been closed.")
    else:
        print("[CloseBox] Task completed.")

    return obs, reward, done
