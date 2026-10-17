


    sponge_object = Shape('sponge')
    sponge_position = np.array(sponge_object.get_position(), dtype=np.float64)
    target_z = float(getattr(task._task, 'z_boundary', sponge_position[2]))
    travel_z = target_z + 0.03
    wipe_z = target_z + 0.0175
    dirt_spots = getattr(task._task, 'dirt_spots', [])
    dirt_xy = np.array([np.array(d.get_position(), dtype=np.float64)[:2] for d in dirt_spots])

    (x_min, y_min) = dirt_xy.min(axis=0)
    (x_max, y_max) = dirt_xy.max(axis=0)
    x_min -= 0.075
    x_max += 0.075
    y_min -= 0.04
    y_max += 0.04

    (obs, reward, done) = sawyer_pick(env, task, target_object=sponge_object, target_pos=sponge_position.tolist(), grasp_offset=[0.0, 0.0, 0.0])
    if done:
        return (obs, reward, done)

    row_step = 0.04
    ys = np.arange(y_max, y_min + 1e-09, -row_step)

    (obs, reward, done) = sawyer_move_to(env, task, target_pos=[float(x_min), float(ys[0]), float(travel_z)], timeout_s=10.0)
    if done:
        return (obs, reward, done)

    for (i, y) in enumerate(ys):
        y = float(y)
        (x0, x1) = (float(x_min), float(x_max)) if i % 2 == 0 else (float(x_max), float(x_min))

        eps = 0.01
        dir_sign = 1.0 if x1 - x0 > 0 else -1.0
        x1_pre = float(x1 - dir_sign * eps)

        (obs, reward, done) = sawyer_move_to(env, task, target_pos=[x0, y, float(travel_z)], timeout_s=10.0)
        if done:
            return (obs, reward, done)

        (