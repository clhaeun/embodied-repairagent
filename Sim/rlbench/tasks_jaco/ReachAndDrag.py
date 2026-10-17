import numpy as np
from time import sleep
from pyrep.objects.shape import Shape
from scipy.spatial.transform import Rotation as R

from skill_code_jaco import (
    jaco_move_linear,
    jaco_rotate_gripper,
    jaco_align_gripper,
    jaco_close_gripper
)


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    Test Jaco rotation skill using ReachAndDrag task.
    Reach the stick and drag it using gripper rotation.
    """
    sleep(1.0)

    stick = Shape('stick')
    stick_pos = np.array(stick.get_position(), dtype=float)
    stick_quat = np.array(stick.get_quaternion(), dtype=float)

    print("[ReachAndDrag] Aligning gripper for grasping stick...")
    obs, reward, done = jaco_align_gripper(
        env, task,
        approach_direction="down",
        maintain_position=True,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    approach_pos = stick_pos + np.array([0.0, 0.0, 0.15], dtype=float)
    print(f"[ReachAndDrag] Moving to stick approach: {approach_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=approach_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print(f"[ReachAndDrag] Rotating gripper to match stick orientation...")
    obs, reward, done = jaco_rotate_gripper(
        env, task,
        target_quat=stick_quat,
        maintain_position=True,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print(f"[ReachAndDrag] Moving down to stick: {stick_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=stick_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[ReachAndDrag] Closing gripper to grasp stick...")
    obs, reward, done = jaco_close_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    drag_pos = stick_pos + np.array([0.15, 0.0, 0.05], dtype=float)
    print(f"[ReachAndDrag] Dragging stick: {drag_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=drag_pos,
        n_waypoints=12,
        timeout_s=10.0
    )

    if done:
        print("[ReachAndDrag] Success! Stick has been dragged.")
    else:
        print("[ReachAndDrag] Task completed.")

    return obs, reward, done
