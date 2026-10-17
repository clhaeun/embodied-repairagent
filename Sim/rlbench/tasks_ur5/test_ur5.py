# test_skill_code_ur5.py
#
# 목적:
# - skill_code_ur5.py에 있는 UR5 스킬(ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper)을
#   하나씩 독립적으로 실행해보는 테스트 러너.
#
# 주의:
# - RLBench task는 성공(done=True)하면 환경이 꺼질 수 있으므로,
#   "한 프로세스에서 여러 스킬을 연속으로 길게 테스트"하기보단,
#   한 번의 run_skill 호출에서 "테스트 1개"만 수행하거나,
#   각 테스트를 별도의 에피소드로 분리하는 구성이 안전함.
#
# - object는 참고 코드에 나온 것만 사용: Shape('ball'), Shape('basket_ball_hoop_respondable')
#
# 사용:
# - RLBench 쪽에서 tasks/basketball_in_hoop.py가 run_skill(env, task, ...)을 호출하듯
#   이 파일의 run_skill을 task에 연결해서 테스트.
#
# - 스킬은 skill_code_ur5.py에서 import 하도록 구성.

from time import sleep
import numpy as np

from pyrep.objects.shape import Shape

# UR5 skills
from skill_code_ur5 import (
    ur5_move_to,
    ur5_grasp_at,
    ur5_release_at,
    ur5_align_gripper,
)

# ------------------------------------------------------------
# Helper: safe "done" handling
# ------------------------------------------------------------

def _step_guard(task):
    """현재 task가 이미 success 상태면 바로 return."""
    try:
        success, _terminate = task._task.success()
        return bool(success)
    except Exception:
        return False


# ------------------------------------------------------------
# One-by-one tests
# ------------------------------------------------------------

def _test_move_only(env, task):
    """
    Move만 단독 테스트:
    - 공을 건드리지 않는 범위에서 "공 위로 살짝 이동" 정도로 테스트.
    """
    ball = Shape('ball')
    ball_pos = np.array(ball.get_position(), dtype=float)

    # 공 위로 충분히 띄운 지점 (top-down 충돌 회피)
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
    q = np.array(ball.get_quaternion(), dtype=float)  # xyzw

    print(f"[TEST] ur5_align_gripper (reference=ball quat) quat={q}")
    obs, reward, done = ur5_align_gripper(
        env, task,
        reference_quat=q,
        approach_direction="down",   # top-down 기준
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
        grasp_quat=None,  # 기존 동작 유지: 현재 그리퍼 orientation 유지
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

    # hoop 옆/위로: 기존 참고 코드처럼 x방향 접근 형태
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

    # 1) grasp
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

    # 2) move/release above hoop (x-approach 형태)
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

    # 3) align (ball quat 기반)
    q = np.array(ball.get_quaternion(), dtype=float)
    print(f"[SEQ] ur5_align_gripper reference quat={q}")
    obs, reward, done = ur5_align_gripper(
        env, task,
        reference_quat=q,
        approach_direction="right",   # 참고 코드에서 approach_direction='x' 느낌 -> 'right'가 의미상 가까움
        yaw_mode="parallel",
        timeout_s=10.0
    )
    return obs, reward, done


# ------------------------------------------------------------
# Main entry used by RLBench task runner
# ------------------------------------------------------------

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    """
    테스트 모드 선택:
    - 아래 TEST_MODE를 바꿔가며 '스킬 하나씩' 테스트하기.
    - task done 되면 환경이 꺼질 수 있으므로, 기본은 "단일 스킬 테스트" 권장.
    """
    sleep(1.0)

    TEST_MODE = "single"  # "single" or "sequence"
    SINGLE_WHICH = "grasp"  # "move" | "align" | "grasp" | "release"

    # 혹시 이미 성공 상태면 바로 종료
    if _step_guard(task):
        return

    if TEST_MODE == "sequence":
        _test_full_sequence(env, task)
        return

    # single
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