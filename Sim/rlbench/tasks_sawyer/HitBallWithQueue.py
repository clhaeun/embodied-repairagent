 queue_object = Shape('queue')
     ball_object = Shape('ball')
     queue_position = queue_object.get_position()
     ball_position = ball_object.get_position()
     (queue_sizes, _) = get_bbox_sizes(queue_object)
     l = float(np.max(queue_sizes)) / 2.0
     (ball_sizes, _) = get_bbox_sizes(ball_object)
     r = float(np.max(ball_sizes)) / 2.0
     hover_pos = queue_position.copy()
     hover_pos[2] += 0.05
     obs, reward, done = sawyer_move_to(env, task, target_pos=hover_pos, timeout_s=10.0)
     gripper_quat = normalize_vector(obs.gripper_pose[3:7])
     queue_quat = normalize_vector(queue_object.get_quaternion())
     queue_y = R.from_quat(queue_quat).apply([0.0, 1.0, 0.0])
     queue_y = queue_y / np.linalg.norm(queue_y)
     yaw = np.arctan2(queue_y[1], queue_y[0])
     yaw = 0.5 * yaw
     q_yaw = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float64)
     target_quat = normalize_vector(quat_mul(q_yaw, gripper_quat))
     obs, reward, done = sawyer_pick(env, task, target_object=queue_object, target_pos=queue_position, grasp_offset=[0.0, 0.0, 0.0], timeout_s=10.0)
     queue_pos = queue_position.copy()
     queue_pos[1] -= l + r + 0.08
     queue_pos[2] = 0.77
     obs, reward, done = sawyer_move_to(env, task, target_pos=queue_pos, timeout_s=10.0)
     queue_pos[0] += ball_position[0] - queue_position[0] + r
     obs, reward, done = sawyer_move_to(env, task, target_pos=queue_pos, timeout_s=10.0)
     target_pos = ball_position.copy()
     target_pos[1] -= 0.