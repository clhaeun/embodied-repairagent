    ball = Shape('ball')
    hoop = Shape('basket_ball_hoop_respondable')
    success_sensor = ProximitySensor('success')
    ball_pos = ball.get_position()
    hoop_pos = hoop.get_position()
    success_pos = success_sensor.get_position()


    before_place_pos = hoop_pos + [-0.2, 0.0, 0.2]


    target_pos = hoop_pos + [0.0, 0.0, 0.15]


    obs, reward, done = pick(env, task, target_pos=ball_pos, approach_axis='z', approach_distance=0.1, timeout=5.0)


    obs, reward, done = move(env, task, target_pos=before_place_pos, timeout=5.0)


    obs, reward, done = place(env, task, target_pos=target_pos, approach_axis='z', approach_distance=0.1, timeout=5.0)


    if done:
        print("Success! Ball has been placed in the hoop.")
    else:
        print("Failed! Ball was not placed in the hoop.")

    return obs, reward, done