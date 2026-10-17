import numpy as np
from time import sleep
from pyrep.objects.joint import Joint

from skill_code_jaco import (
    jaco_move_linear,
    jaco_align_gripper
)


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    Test Jaco alignment and linear movement using PushButton task.
    Align gripper properly and push the button.
    """
    sleep(1.0)

    button = Joint('target_button_joint')
    button_joint_pos = np.array(button.get_position(), dtype=float)

    print("[PushButton] Aligning gripper down for pushing...")
    obs, reward, done = jaco_align_gripper(
        env, task,
        approach_direction="down",
        maintain_position=True,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    approach_pos = button_joint_pos + np.array([0.0, 0.0, 0.1], dtype=float)
    print(f"[PushButton] Moving to approach position: {approach_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=approach_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    push_pos = button_joint_pos + np.array([0.0, 0.0, -0.03], dtype=float)
    print(f"[PushButton] Pushing button: {push_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=push_pos,
        n_waypoints=8,
        timeout_s=8.0
    )

    if done:
        print("[PushButton] Success! Button has been pushed.")
    else:
        print("[PushButton] Task completed.")

    return obs, reward, done
