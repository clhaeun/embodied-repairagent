    phone = Shape('phone')
    phone_pos = phone.get_position()
    success = ProximitySensor('success')
    success_pos = success.get_position()

    above_place_pos = success_pos + [0.0, 0.0, 0.2]
    obs, reward, done = sawyer_pick(env, task, target_object=phone, target_pos=phone_pos, grasp_offset=[0.0, 0.0, 0.0])

    if done:
        obs, reward, done = sawyer_place(env, task, place_pos=above_place_pos, place_offset=[0.0, 0.0, 0.0])

    place_pos = success_pos + [0.0, 0.0, 0.05]
    obs, reward, done = sawyer_place(env, task, place_pos=place_pos, place_offset=[0.0, 0.0, 0.0])

    if done:
        print("Success! Phone has been placed on the base.")
    else:
        print("Failed! Phone was not placed on the base.")

    return obs, reward, done