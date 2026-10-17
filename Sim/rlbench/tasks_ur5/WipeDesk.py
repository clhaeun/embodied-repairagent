```python
def run_skill(env, task, descriptions=None, obs=None, variation_index: int = 1):
    sponge_object = Shape('sponge')
    sponge_position = np.array(sponge_object.get_position(), dtype=np.float64)

    target_z = float(getattr(task._task, "z_boundary", sponge_position[2]))
    travel_z = target_z + 0.03
    wipe_z = target_z + 0.0175

    dirt_spots = getattr(task._task, "dirt_spots", [])
    dirt_xy = np.array([np.array(d.get_position(), dtype=np.float64)[:2] for d in dirt_spots])
    
    x_min, y_min = dirt_xy.min(axis=0)
    x_max, y_max = dirt_xy.max(axis=0)

    x_min -= 0.075
    x_max += 0.075
    y_min -= 0.04
    y_max += 0.04

    above_grasp_pos = sponge_position + [0.0, 0.0, 0.15]
    q = sponge_object.get_quaternion()
    obs, reward, done = ur5_move_to(env, task, target_pos=above_grasp_pos.tolist(), timeout_s=10.0)
    if done:
        return obs, reward, done

    obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction='down', timeout_s=10.0)
    if done:
        return obs, reward, done

    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=sponge_position.tolist(), approach={'axis': 'z', 'distance': 0.15}, timeout_s=10.0)
    if done:
        return obs, reward, done

    row_step = 0.04
    ys = np.arange(y_max, y_min + 1e-9, -row_step)

    obs, reward, done = ur5_move_to(env, task, target_pos=[float(x_min), float(ys[0]), float(travel_z)], timeout_s=10.0)
    if done:
        return obs, reward, done

    for i, y in enumerate(ys):
        y = float(y)
        x0, x1 = (float(x_min), float(x_max)) if i % 2 == 0 else (float(x_max), float(x_min))
        
        eps = 0.01
        dir_sign = 1.0 if (x1 - x0) > 0 else -1.0
        x1_pre = float(x1 - dir_sign * eps)

        obs, reward, done = ur5_move_to(env, task, target_pos=[x0, y, float(travel_z)], timeout_s=10.0)
        if done:
            return obs, reward, done
            
        obs, reward, done = ur5_move_to(env, task, target_pos=[x0, y, float(wipe_z)], timeout_s=10.0)
        if done:
            return obs, reward, done
            
        obs, reward, done = ur5_move_to(env, task, target_pos=[x1_pre, y, float(wipe_z)], timeout_s=15.0)
        if done:
            return obs, reward, done

        obs, reward, done = ur5_move_to(env, task, target_pos=[x1, y, float(wipe_z)], timeout_s=15.0)
        if done:
            return obs, reward, done
            
        obs, reward, done = ur5_move_to(env, task, target_pos=[x1, y, float(travel_z)], timeout_s=10.0)
        if done:
            return obs, reward, done

    return obs, reward, done
```