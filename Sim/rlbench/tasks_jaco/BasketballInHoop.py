import numpy as np
from time import sleep
from pyrep.objects.shape import Shape

from skill_code_jaco import (
    jaco_move_arc,
    jaco_align_gripper,
    jaco_open_gripper,
    jaco_close_gripper,
    jaco_move_linear
)


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    Test Jaco arc movement skill using BasketballInHoop task.
    Pick up the ball and throw it into the hoop using arc trajectory.
    """
    sleep(1.0)

    ball = Shape('ball')
    hoop = Shape('basket_ball_hoop_respondable')

    ball_pos = np.array(ball.get_position(), dtype=float)
    hoop_pos = np.array(hoop.get_position(), dtype=float)

    print("[BasketballInHoop] Aligning gripper down for picking ball...")
    obs, reward, done = jaco_align_gripper(
        env, task,
        approach_direction="down",
        maintain_position=True,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[BasketballInHoop] Opening gripper...")
    obs, reward, done = jaco_open_gripper(
        env, task,
        maintain_pose=True,
        n_steps=8,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    approach_pos = ball_pos + np.array([0.0, 0.0, 0.15], dtype=float)
    print(f"[BasketballInHoop] Moving to ball approach position: {approach_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=approach_pos,
        n_waypoints=10,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print(f"[BasketballInHoop] Moving down to ball: {ball_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=ball_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    print("[BasketballInHoop] Closing gripper to grasp ball...")
    obs, reward, done = jaco_close_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )
    if done:
        return obs, reward, done

    lift_pos = ball_pos + np.array([0.0, 0.0, 0.15], dtype=float)
    print(f"[BasketballInHoop] Lifting ball: {lift_pos}")
    obs, reward, done = jaco_move_linear(
        env, task,
        target_pos=lift_pos,
        n_waypoints=8,
        timeout_s=8.0
    )
    if done:
        return obs, reward, done

    release_pos = hoop_pos + np.array([0.1, 0.0, 0.2], dtype=float)
    print(f"[BasketballInHoop] Moving to hoop with ARC trajectory: {release_pos}")
    obs, reward, done = jaco_move_arc(
        env, task,
        target_pos=release_pos,
        arc_height=0.15,
        n_waypoints=20,
        timeout_s=12.0
    )
    if done:
        return obs, reward, done

    print("[BasketballInHoop] Opening gripper to release ball...")
    obs, reward, done = jaco_open_gripper(
        env, task,
        maintain_pose=True,
        n_steps=10,
        timeout_s=5.0
    )

    if done:
        print("[BasketballInHoop] Success! Ball has been thrown.")
    else:
        print("[BasketballInHoop] Task completed.")

    return obs, reward, done
