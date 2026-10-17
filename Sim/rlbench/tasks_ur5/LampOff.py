    button = Joint('target_button_joint')
    button_joint_pos = button.get_position()

    # Define the target position for pushing the button
    button_push_pos = button_joint_pos + [0.0, 0.0, 0.0]

    # Perform the push action
    obs, reward, done = push(env, task, target_pos=button_push_pos, timeout=10.0)

    # Check if the button was successfully pushed
    if done:
        print("Success! Button has been pushed.")
    else:
        print("Failed! Button was not pushed.")

    return obs, reward, done