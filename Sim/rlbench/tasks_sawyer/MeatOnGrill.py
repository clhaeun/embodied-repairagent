
    chicken = Shape('chicken')
    success = ProximitySensor('success')

    chicken_pos = chicken.get_position()
    success_pos = success.get_position()


    pick_pos = success_pos + [0.0, 0.0, 0.0]
    obs, reward, done = sawyer_pick(env, task, target_object=chicken, target_pos=pick_pos, grasp_offset=[0.0, 0.0, 0.0])


    above_place_pos = success_pos + [0.0, 0.0, 0.2]
    obs, reward, done = sawyer_move_to(env, task, target_pos=above_place_pos, timeout_s=10.0)


    target_place_pos = success_pos + [0.0, 0.0, 0.1]
    obs, reward, done = sawyer_place(env, task, place_pos=target_place_pos, timeout_s=10.0)

    return obs, reward, done
```