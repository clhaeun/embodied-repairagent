    button = Joint('target_button_joint')
    button_joint_pos = button.get_position()

    button_push_pos = button_joint_pos + [0.0, 0.0, -0.03]
    obs, reward, done = sawyer_pick(env, task, target_object='push_button_target', target_pos=button_push_pos, grasp_offset=[0.0, 0.0, 0.0])

    if done:
        obs, reward, done = sawyer_place(env, task, place_pos=button_push_pos, place_offset=[0.0, 0.0, 0.0])

    return obs, reward, done