# tasks/wipe_desk.py - done
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    print(f"\n[DEBUG] Starting wipe_desk task (variation {variations_index})")
    
    sponge_object = Shape('sponge')
    sensor = ProximitySensor('sponge_sensor')
    sponge_position = np.array(sponge_object.get_position(), dtype=np.float64)
    print(f"[DEBUG] Sponge position: {sponge_position}")

    # dirt height, travel height, wipe height
    target_z = float(getattr(task._task, "z_boundary", sponge_position[2]))
    travel_z = target_z + 0.03
    wipe_z = target_z + 0.0175
    print(f"[DEBUG] Heights - target_z: {target_z:.4f}, travel_z: {travel_z:.4f}, wipe_z: {wipe_z:.4f}")

    # dirt_spots boundary
    dirt_spots = getattr(task._task, "dirt_spots", [])
    initial_dirt_count = len(dirt_spots)
    print(f"[DEBUG] Number of dirt spots: {initial_dirt_count}")
    
    dirt_xy = []
    for d in dirt_spots:
        try:
            p = np.array(d.get_position(), dtype=np.float64)
            dirt_xy.append(p[:2])
        except Exception as e:
            print(f"[DEBUG] Failed to get dirt spot position: {e}")
            pass

    dirt_xy = np.stack(dirt_xy, axis=0)
    x_min, y_min = dirt_xy.min(axis=0)
    x_max, y_max = dirt_xy.max(axis=0)
    print(f"[DEBUG] Dirt boundary (before margin) - x: [{x_min:.4f}, {x_max:.4f}], y: [{y_min:.4f}, {y_max:.4f}]")

    # 마진(조금 크게 닦기)
    x_min -= 0.075
    x_max += 0.075
    y_min -= 0.04
    y_max += 0.04
    print(f"[DEBUG] Dirt boundary (after margin) - x: [{x_min:.4f}, {x_max:.4f}], y: [{y_min:.4f}, {y_max:.4f}]")

    # Pick sponge
    print(f"[DEBUG] Picking sponge at position: {sponge_position.tolist()}")
    obs, reward, done = pick(env, task, target_pos=sponge_position.tolist(), approach_distance=0.15, approach_axis='z', timeout=20.0)
    print(f"[DEBUG] After pick - reward: {reward:.4f}, done: {done}")
    if done:
        print("[DEBUG] Task completed after picking sponge")
        return obs, reward, done

    # 지그재그 경로 생성
    row_step = 0.04
    ys = np.arange(y_max, y_min + 1e-9, -row_step)
    print(f"[DEBUG] Generated {len(ys)} rows for wiping (row_step: {row_step})")

    # travel_z
    print(f"[DEBUG] Moving to starting position: [{x_min:.4f}, {ys[0]:.4f}, {travel_z:.4f}]")
    obs, reward, done = move(env, task, target_pos=[float(x_min), float(ys[0]), float(travel_z)], timeout=20.0)
    print(f"[DEBUG] After initial move - reward: {reward:.4f}, done: {done}")
    if done:
        print("[DEBUG] Task completed during initial move")
        return obs, reward, done

    # Wipe
    for i, y in enumerate(ys):
        y = float(y)
        print(f"\n[DEBUG] Row {i+1}/{len(ys)} at y={y:.4f}")

        # 지그재그
        if i % 2 == 0: 
            x0, x1 = float(x_min), float(x_max)
            direction = "left to right"
        else: 
            x0, x1 = float(x_max), float(x_min)
            direction = "right to left"
        
        print(f"[DEBUG] Direction: {direction}, x0={x0:.4f}, x1={x1:.4f}")
        
        eps = 0.01  # 2cm 정도. 필요하면 0.01~0.03 튜닝
        dir_sign = 1.0 if (x1 - x0) > 0 else -1.0
        x1_pre = float(x1 - dir_sign * eps)
        print(f"[DEBUG] x1_pre={x1_pre:.4f} (eps={eps})")

        # Wipe 시작점
        print(f"[DEBUG] Moving to row start (travel_z): [{x0:.4f}, {y:.4f}, {travel_z:.4f}]")
        obs, reward, done = move(env, task, target_pos=[x0, y, float(travel_z)], timeout=20.0)
        print(f"[DEBUG] After move to start - reward: {reward:.4f}, done: {done}")
        if done: 
            print(f"[DEBUG] Task completed at row {i+1} start")
            return obs, reward, done
            
        # wipe_z까지 내려가기
        print(f"[DEBUG] Lowering to wipe_z: [{x0:.4f}, {y:.4f}, {wipe_z:.4f}]")
        obs, reward, done = move(env, task, target_pos=[x0, y, float(wipe_z)], timeout=20.0)
        print(f"[DEBUG] After lowering - reward: {reward:.4f}, done: {done}")
        if done: 
            print(f"[DEBUG] Task completed while lowering at row {i+1}")
            return obs, reward, done
            
        # x방향으로 쭉 닦기
        print(f"[DEBUG] Wiping to x1_pre: [{x1_pre:.4f}, {y:.4f}, {wipe_z:.4f}]")
        obs, reward, done = move(env, task, target_pos=[x1_pre, y, float(wipe_z)], timeout=25.0)
        print(f"[DEBUG] After wiping to x1_pre - reward: {reward:.4f}, done: {done}")
        if done: 
            print(f"[DEBUG] Task completed while wiping to x1_pre at row {i+1}")
            return obs, reward, done

        print(f"[DEBUG] Wiping to x1: [{x1:.4f}, {y:.4f}, {wipe_z:.4f}]")
        obs, reward, done = move(env, task, target_pos=[x1, y, float(wipe_z)], timeout=25.0)
        print(f"[DEBUG] After wiping to x1 - reward: {reward:.4f}, done: {done}")
        if done: 
            print(f"[DEBUG] Task completed while wiping to x1 at row {i+1}")
            return obs, reward, done
            
        # 다시 travel_z로
        print(f"[DEBUG] Raising to travel_z: [{x1:.4f}, {y:.4f}, {travel_z:.4f}]")
        obs, reward, done = move(env, task, target_pos=[x1, y, float(travel_z)], timeout=20.0)
        print(f"[DEBUG] After raising - reward: {reward:.4f}, done: {done}")

        # 남은 dirt 개수 확인
        # remaining_dirt = len(task._task.dirt_spots)
        # cleaned_dirt = initial_dirt_count - remaining_dirt
        # print(f"[DEBUG] Row {i+1} complete - Remaining dirt: {remaining_dirt}/{initial_dirt_count}, Cleaned: {cleaned_dirt}")

        if done:
            print(f"[DEBUG] Task completed after raising at row {i+1}")
            return obs, reward, done

    if done:
        print("\n[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("\n[Task] Task not completed yet (done=False).")
        return obs, reward, done