# tasks/wipe_desk.py - done
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, quat_mul, normalize_vector, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    sponge_object = Shape('sponge')
    sensor = ProximitySensor('sponge_sensor')
    sponge_position = np.array(sponge_object.get_position(), dtype=np.float64)

    # dirt height, travel height, wipe height
    target_z = float(getattr(task._task, "z_boundary", sponge_position[2]))
    travel_z = target_z + 0.03
    wipe_z = target_z + 0.0175

    # dirt_spots boundary
    dirt_spots = getattr(task._task, "dirt_spots", [])
    dirt_xy = []
    for d in dirt_spots:
        try:
            p = np.array(d.get_position(), dtype=np.float64)
            dirt_xy.append(p[:2])
        except Exception:
            pass

    dirt_xy = np.stack(dirt_xy, axis=0)
    x_min, y_min = dirt_xy.min(axis=0)
    x_max, y_max = dirt_xy.max(axis=0)

    # 마진(조금 크게 닦기)
    x_min -= 0.075
    x_max += 0.075
    y_min -= 0.04
    y_max += 0.04

    # Pick sponge
    obs, reward, done = pick(env, task, target_pos=sponge_position.tolist(), approach_distance=0.15, approach_axis='z', timeout=10.0)

    # 지그재그 경로 생성
    row_step = 0.04
    ys = np.arange(y_max, y_min + 1e-9, -row_step)

    # travel_z
    obs, reward, done = move(env, task, target_pos=[float(x_min), float(ys[0]), float(travel_z)], timeout=10.0)

    # Wipe
    for i, y in enumerate(ys):
        y = float(y)

        # 지그재그
        if i % 2 == 0: x0, x1 = float(x_min), float(x_max)
        else: x0, x1 = float(x_max), float(x_min)
        eps = 0.02  # 2cm 정도. 필요하면 0.01~0.03 튜닝
        dir_sign = 1.0 if (x1 - x0) > 0 else -1.0
        x1_pre = float(x1 - dir_sign * eps)

        # Wipe 시작점
        obs, reward, done = move(env, task, target_pos=[x0, y, float(travel_z)], timeout=10.0)
        if done: return obs, reward, done
        # wipe_z까지 내려가기
        obs, reward, done = move(env, task, target_pos=[x0, y, float(wipe_z)], timeout=10.0)
        if done: return obs, reward, done
        # x방향으로 쭉 닦기
        obs, reward, done = move(env, task, target_pos=[x1_pre, y, float(wipe_z)], timeout=15.0)
        if done: return obs, reward, done

        obs, reward, done = move(env, task, target_pos=[x1, y, float(wipe_z)], timeout=15.0)
        if done: return obs, reward, done 
        # 다시 travel_z로
        obs, reward, done = move(env, task, target_pos=[x1, y, float(travel_z)], timeout=10.0)
        if done: return obs, reward, done

    if done:
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")