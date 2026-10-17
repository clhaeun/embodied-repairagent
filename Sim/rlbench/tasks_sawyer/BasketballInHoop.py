    ball = Shape('ball')
    hoop = Shape('basket_ball_hoop_respondable')
    ball_stop = Shape('ball_stop')

    ball_pos = ball.get_position()
    hoop_pos = hoop.get_position()
    ball_stop_pos = ball_stop.get_position()

    # Define the target position for moving before placing
    before_place_pos = hoop_pos + [-0.2, 0.0, 0.2]

    # Define the target position for placing the ball into the hoop
    target_pos = hoop_pos + [0.0, 0.0, 0.15]

    # Step 1: Pick up the ball
    obs, reward, done = sawyer_pick(env, task, target_object=ball, target_pos=ball_pos, grasp_offset=[0.0, 0.0, 0.0])

    # Step 2: Move before place the ball
    obs, reward, done = sawyer_move_to(env, task, target_pos=before_place_pos, timeout_s=30.0)

    # Step 3: Place the ball into the hoop
    obs, reward, done = sawyer_place(env, task, place_pos=target_pos, place_offset=[0.0, 0.0, 0.0])

    # Step 4: Check if the ball has successfully entered the hoop using the ball stop sensor
    if done:
        print("Success! Ball has been placed in the hoop.")
    else:
        print("Failed! Ball was not placed in the hoop.")

    return obs, reward, done