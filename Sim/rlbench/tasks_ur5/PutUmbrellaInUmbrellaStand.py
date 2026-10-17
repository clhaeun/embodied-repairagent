q = umbrella.get_quaternion()
obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction='right', timeout_s=10.0)
safe_hover_pos = stand_position + [0.0, 0.0, 0.3]
obs, reward, done = ur5_move_to(env, task, target_pos=safe_hover_pos, timeout_s=10.0)
release_pos = safe_hover_pos + [0.0, 0.0, -0.1]
obs, reward, done = ur5_release_at(env, task, place_pos=release_pos, approach={'axis':'z', 'distance':0.1}, timeout_s=10.0)
return (obs, reward, done)