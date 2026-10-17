    charger = Shape('charger')
    charger_pos = charger.get_position()

    pick_pos = charger_pos + [0.0, 0.0, 0.0]
    obs, reward, done = sawyer_pick(env, task, target_object=charger, target_pos=pick_pos, grasp_offset=[0.0, 0.0, 0.0])

    unplug_pos = charger_pos + [-0.2, 0.0, 0.0]
    obs, reward, done = sawyer_place(env, task, place_pos=unplug_pos, place_offset=[0.0, 0.0, 0.0])

    if done:
        print("Success! Charger has been unplugged.")
    else:
        print("Failed! Charger was not unplugged.")

    return obs, reward, done