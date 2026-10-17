    button = Joint('target_button_joint')
    button_joint_pos = button.get_position()


    button_push_pos = button_joint_pos + [0.0, 0.0, 0.0]


    obs, reward, done = push(env, task, target_pos=button_push_pos, timeout=10.0)


    if done:
        print("Success! Button has been pushed.")
    else:
        print("Failed! Button was not pushed.")

    return obs, reward, done