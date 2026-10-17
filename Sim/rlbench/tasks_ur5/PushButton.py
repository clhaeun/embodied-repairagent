    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=button_joint_pos, approach={'axis': 'z', 'distance': 0.1}, timeout_s=10.0)
    if done:
        print("Success! Button has been pushed.")
    else:
        print("Failed! Button was not pushed.")
    return obs, reward, done