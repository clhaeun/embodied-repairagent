    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=button_joint_pos, approach=button_approach_direction, timeout_s=button_timeout)
    obs, reward, done = ur5_release_at(env, task, place_pos=button_push_pos, approach=button_approach_direction, timeout_s=button_timeout)
    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=lampshade_pos, approach=lampshade_approach_direction, timeout_s=lampshade_timeout)
    obs, reward, done = ur5_release_at(env, task, place_pos=lampshade_grasp_pos, approach=lampshade_approach_direction, timeout_s=lampshade_timeout)
    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=lamp_screw_pos, approach=lamp_screw_approach_direction, timeout_s=lamp_screw_timeout)
    obs, reward, done = ur5_release_at(env, task, place_pos=lamp_screw_grasp_pos, approach=lamp_screw_approach_direction, timeout_s=lamp_screw_timeout)
    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=bulb_pos, approach=