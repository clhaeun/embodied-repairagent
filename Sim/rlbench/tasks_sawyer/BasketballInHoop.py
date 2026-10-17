    ball = Shape('ball')
    hoop = Shape('basket_ball_hoop_respondable')
    ball_stop = Shape('ball_stop')

    ball_pos = ball.get_position()
    hoop_pos = hoop.get_position()
    ball_stop_pos = ball_stop.get_position()


    before_place_pos = hoop_pos + [-0.2, 0.0, 0.2]


    target_pos = hoop_pos + [0.0, 0.0, 0.15]


    obs, reward, done = sawyer_pick(env, task, target_object=ball, target_pos=ball_pos, grasp_offset=[0.0, 0.0, 0.0])


    obs, reward, done = sawyer_move_to(env, task, target_pos=before_place_pos, timeout_s=30.0)


    obs, reward, done = sawyer_place(env, task, place_pos=target_pos, place_offset=[0.0, 0.0, 0.0])


    if done:
        print("Success! Ball has been placed in the hoop.")
    else:
        print("Failed! Ball was not placed in the hoop.")

    return obs, reward, done