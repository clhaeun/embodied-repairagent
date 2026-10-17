    charger = Shape('charger')
    charger_pos = charger.get_position()

    # Define the target positions for pick and place
    pick_pos = charger_pos + [0.0, 0.0, 0.0]
    place_pos = charger_pos + [-0.2, 0.0, 0.0]

    # Unplug the charger
    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=pick_pos, approach={'axis':'z', 'distance':0.15}, timeout_s=10.0)
    obs, reward, done = ur5_release_at(env, task, place_pos=place_pos, approach={'axis':'x', 'distance':0.1}, timeout_s=10.0)

    # Check if the charger was successfully unplugged
    if done:
        print("Success! Charger has been unplugged.")
    else:
        print("Failed! Charger was not unplugged.")

    return obs, reward, done