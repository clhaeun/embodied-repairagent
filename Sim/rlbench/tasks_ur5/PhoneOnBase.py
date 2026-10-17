    (obs, reward, done) = ur5_release_at(env, task, place_pos=success_pos, timeout_s=5.0)
    if done:
        print("Success! Phone has been placed on the base.")
    else:
        print("Failed! Phone was not placed on the base.")
    return obs, reward, done