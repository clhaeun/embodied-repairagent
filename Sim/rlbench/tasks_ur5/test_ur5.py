



















from time import sleep
import numpy as np

from pyrep.objects.shape import Shape


from skill_code_ur5 import (
    ur5_move_to,
    ur5_grasp_at,
    ur5_release_at,
    ur5_align_gripper,
)





def _step_guard(task):
    """현재 task가 이미 success 상태면 바로 return."""
    try:
        success, _terminate = task._task.success()
        return bool(success)
    except Exception:
        return False






def _test_move_only(env, task):
    """
    Move만 단독 테스트:
    - 공을 건드리지 않는 범위에서 "공 위로 살짝 이동" 정도로 테스트.
    """
    ball = Shape('ball')
    ball_pos = np.array(ball.get_position(), dtype=float)


    target = ball_pos + np.array([0.0, 0.0, 0.25], dtype=float)

    print("[TEST] ur5_move_to -> ball above")
    obs, reward, done = ur5_move_to(
        env, task,
        target_pos=target,
        target_quat=None,
        gripper=None,
        n_waypoints=10,
        pos_tol=0.01,
        max_steps_per_wp=100,
        timeout_s=10.0,
        log_cam=None
    )
    return obs, reward, done


def _test_align_only(env, task):
    """
    Align만 단독 테스트:
    - 공의 quaternion을 reference로 yaw를 따라가도록 정렬.
    - 위치는 현재 위치 유지.
    """
    ball = Shape('ball')
    q = np.array(ball.get_quaternion(), dtype=float)

    print(f"[TEST] ur5_align_gripper (reference=ball quat) quat={q}")
    obs, reward, done = ur5_align_gripper(
        env, task,
        reference_quat=q,
        approach_direction="down",
        yaw_mode="parallel",
        yaw_offset_rad=0.0,
        pos_hold=None,
        gripper=None,
        ori_tol=1e-3,
        max_steps=100,
        timeout_s=10.0
    )
    return obs, reward, done


def _test_grasp_only(env, task):
    """
    Grasp만 단독 테스트:
    - 공 위치로 접근해서 집기까지 수행
    """
    ball = Shape('ball')
    ball_pos = np.array(ball.get_position(), dtype=float)

    print("[TEST] ur5_grasp_at -> grasp ball")
    obs, reward, done = ur5_grasp_at(
        env, task,
        grasp_pos=ball_pos,
        grasp_quat=None,
        approach={"axis": "z", "distance": 0.15, "n_waypoints_approach": 5, "n_waypoints_descend": 10},
        gripper_close=-1.0,
        settle_steps=5,
        close_steps=10,
        pos_tol=0.01,
        max_steps_per_wp=100,
        timeout_s=10.0,
        log_cam="cam_front",
        log_prefix="grasp_test"
    )
    return obs, reward, done


def _test_release_only(env, task):
    """
    Release만 단독 테스트:
    - (공을 들고 있다는 가정이 필요하지만) 단독 호출 테스트를 위해
      hoop 근처 위로 이동 후 오픈까지 수행.
    - 공을 들고 있지 않아도 동작 자체는 테스트 가능.
    """
    hoop = Shape('basket_ball_hoop_respondable')
    hoop_pos = np.array(hoop.get_position(), dtype=float)


    place_pos = hoop_pos + np.array([0.1, 0.0, 0.15], dtype=float)

    print("[TEST] ur5_release_at -> open at above hoop-ish")
    obs, reward, done = ur5_release_at(
        env, task,
        place_pos=place_pos,
        place_quat=None,
        approach={"axis": "x", "distance": 0.10, "n_waypoints_approach": 5, "n_waypoints_descend": 10},
        gripper_open=1.0,
        settle_steps=5,
        open_steps=10,
        pos_tol=0.01,
        max_steps_per_wp=100,
        timeout_s=10.0,
        log_cam="cam_front",
        log_prefix="release_test"
    )
    return obs, reward, done


def _test_full_sequence(env, task):
    """
    참고 코드 흐름을 UR5 스킬로 재작성:
    - grasp(ball)
    - release(at hoop above-ish)  (원래 place를 move+open으로 썼던 흐름을 release로 합침)
    - align(ball quat 기준)
    """
    ball = Shape('ball')
    hoop = Shape('basket_ball_hoop_respondable')
    ball_pos = np.array(ball.get_position(), dtype=float)
    hoop_pos = np.array(hoop.get_position(), dtype=float)


    print("[SEQ] ur5_grasp_at")
    obs, reward, done = ur5_grasp_at(
        env, task,
        grasp_pos=ball_pos,
        grasp_quat=None,
        approach={"axis": "z", "distance": 0.15, "n_waypoints_approach": 5, "n_waypoints_descend": 10},
        timeout_s=10.0,
        log_cam="cam_front",
        log_prefix="seq_grasp"
    )
    if done:
        return obs, reward, done


    target_place = hoop_pos + np.array([0.1, 0.0, 0.15], dtype=float)
    print("[SEQ] ur5_release_at")
    obs, reward, done = ur5_release_at(
        env, task,
        place_pos=target_place,
        place_quat=None,
        approach={"axis": "x", "distance": 0.10, "n_waypoints_approach": 5, "n_waypoints_descend": 10},
        timeout_s=10.0,
        log_cam="cam_front",
        log_prefix="seq_release"
    )
    if done:
        return obs, reward, done


    q = np.array(ball.get_quaternion(), dtype=float)
    print(f"[SEQ] ur5_align_gripper reference quat={q}")
    obs, reward, done = ur5_align_gripper(
        env, task,
        reference_quat=q,
        approach_direction="right",
        yaw_mode="parallel",
        timeout_s=10.0
    )
    return obs, reward, done






def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    테스트 모드 선택:
    - 아래 TEST_MODE를 바꿔가며 '스킬 하나씩' 테스트하기.
    - task done 되면 환경이 꺼질 수 있으므로, 기본은 "단일 스킬 테스트" 권장.
    """
    sleep(1.0)

    TEST_MODE = "single"
    SINGLE_WHICH = "grasp"


    if _step_guard(task):
        return

    if TEST_MODE == "sequence":
        _test_full_sequence(env, task)
        return


    if SINGLE_WHICH == "move":
        _test_move_only(env, task)
    elif SINGLE_WHICH == "align":
        _test_align_only(env, task)
    elif SINGLE_WHICH == "grasp":
        _test_grasp_only(env, task)
    elif SINGLE_WHICH == "release":
        _test_release_only(env, task)
    else:
        raise ValueError(f"Unknown SINGLE_WHICH: {SINGLE_WHICH}")