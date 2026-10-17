import numpy as np
from time import sleep
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor

from skill_code_jaco import (
    jaco_move_linear,
    jaco_align_gripper,
    jaco_open_gripper,
    jaco_close_gripper
)


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    Test Jaco linear movement and gripper skills using PickAndLift task.
    Pick up the target and lift it to the success position.
    """
    sleep(1.0)

    target = Shape('pick_and_lift_target')
    success = ProximitySensor('pick_and_lift_success')

    target_pos = np.array(target.get_position(), dtype=float)
    success_pos = np.array(success.get_position(), dtype=float)

    print("[PickAndLift] Aligning gripper down for picking...")
    obs, reward, done = jaco_align_gripper(
        env, task,
        approach_direction="down",
        maintain_position=True,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[PickAndLift] Opening gripper...")
    obs, reward, done = jaco_open_gripper(
        env, task,
        maintain_pose=True,
        n_steps=8,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    approach_pos = target_pos + np.array([0.0, 0.0, 0.15], dtype=float)
    print(f"[PickAndLift] Moving to approach position: {approach_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=approach_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print(f"[PickAndLift] Moving down to target: {target_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=target_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[PickAndLift] Closing gripper to grasp...")
    obs, reward, done = jaco_close_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    lift_pos = target_pos + np.array([0.0, 0.0, 0.15], dtype=float)
    print(f"[PickAndLift] Lifting object: {lift_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=lift_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    place_pos = success_pos + np.array([0.0, 0.0, 0.05], dtype=float)
    print(f"[PickAndLift] Moving to place position: {place_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=place_pos,
        n_waypoints=12,
        timeout_s=10.0
    )
    if done:
        return obs, reward, done

    print("[PickAndLift] Opening gripper to release...")
    obs, reward, done = jaco_open_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )

    if done:
        print("[PickAndLift] Success! Target has been placed.")
    else:
        print("[PickAndLift] Task completed.")

    return obs, reward, done
