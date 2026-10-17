lid_position[2] += 0.05
hover_pos = lid_position.copy()
hover_pos[1] -= 0.3
hover_pos[2] += 0.1
(obs, reward, done) = ur5_move_to(env, task, target_pos=hover_pos, timeout_s=15.0)
hover_pos[2] -= 0.05
(obs, reward, done) = ur5_move_to(env, task, target_pos=hover_pos, timeout_s=15.0)
(obs, reward, done) = ur5_release_at(env, task, place_pos=lid_position, approach={'axis': 'z', 'distance': 0.05}, timeout_s=15.0)
(obs, reward, done) = ur5_move_to(env, task, target_pos=[0.225, -0.14, 0.967], timeout_s=15.0)
above_grasp_pos = [0.225, -0.14, 1.117]
(obs, reward, done) = ur5_move_to(env, task, target_pos=above_grasp_pos, timeout_s=15.0)
q = box_lid.get_quaternion()
(obs, reward, done) = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction='down', timeout_s=15.0)
(obs, reward, done) = ur5_grasp_at(env, task, grasp_pos=[0.225, -0.14, 0.962], approach={'axis': 'z', 'distance': 0.05}, timeout_s=15.0)
(obs, reward, done) = ur5_move_to(env, task, target_pos=[0.225, -0.14, 1.012], timeout_s=15.0)
(obs, reward, done) = ur5_release_at(env, task, place_pos=[0.225, -0.14, 0.962], approach={'axis': 'z', 'distance': 0.05}, timeout_s=15.0)
(obs, reward, done) = ur5_move_to(env, task, target_pos=[0.225, -0.14, 0.962], timeout_s=15.0)
(obs, reward, done) = ur5_grasp_at(env, task, grasp_pos=[0.225, -0.14, 0.962], approach={'axis': 'z', 'distance': 0.05}, timeout_s=15.0)
hover_pos[1] += 0.4
(obs, reward, done) = ur5_move_to(env, task, target_pos=[0.225, -0.09, 0.962], timeout_s=15.0)
(obs, reward, done) = ur5_release_at(env, task, place_pos=[0.225, -0.14, 0.962], approach={'axis': 'z', 'distance': 0.05}, timeout_s=15.0)