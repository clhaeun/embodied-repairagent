import numpy as np
from time import sleep
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor

from skill_code_jaco import (
    jaco_move_arc,
    jaco_move_linear,
    jaco_align_gripper,
    jaco_open_gripper,
    jaco_close_gripper
)


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    Test Jaco arc movement using PhoneOnBase task.
    Pick up the phone and place it on the base using arc trajectory.
    """
    sleep(1.0)

    phone = Shape('phone')
    base = ProximitySensor('phone_on_base_success')

    phone_pos = np.array(phone.get_position(), dtype=float)
    base_pos = np.array(base.get_position(), dtype=float)

    print("[PhoneOnBase] Aligning gripper down for picking phone...")
    obs, reward, done = jaco_align_gripper(
        env, task,
        approach_direction="down",
        maintain_position=True,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[PhoneOnBase] Opening gripper...")
    obs, reward, done = jaco_open_gripper(
        env, task,
        maintain_pose=True,
        n_steps=8,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    approach_pos = phone_pos + np.array([0.0, 0.0, 0.12], dtype=float)
    print(f"[PhoneOnBase] Moving to phone approach: {approach_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=approach_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print(f"[PhoneOnBase] Moving down to phone: {phone_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=phone_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[PhoneOnBase] Closing gripper to grasp phone...")
    obs, reward, done = jaco_close_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    lift_pos = phone_pos + np.array([0.0, 0.0, 0.12], dtype=float)
    print(f"[PhoneOnBase] Lifting phone: {lift_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=lift_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    place_pos = base_pos + np.array([0.0, 0.0, 0.08], dtype=float)
    print(f"[PhoneOnBase] Moving to base with ARC trajectory: {place_pos}")
    obs, reward, done = jaco_move_arc(
        env, task,
        target_pos=place_pos,
        arc_height=0.12,
        n_waypoints=18,
        timeout_s=12.0
    )
    if done:
        return obs, reward, done

    print("[PhoneOnBase] Opening gripper to release phone...")
    obs, reward, done = jaco_open_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )

    if done:
        print("[PhoneOnBase] Success! Phone has been placed on base.")
    else:
        print("[PhoneOnBase] Task completed.")

    return obs, reward, done
