knife_object = Shape('knife')
chopping_board_object = Shape('chopping_board')
knife_position = knife_object.get_position()
chopping_board_position = chopping_board_object.get_position()
knife_sizes, _ = get_bbox_sizes(knife_object)
l = float(np.max(knife_sizes))

above_grasp_pos = knife_position.copy()
above_grasp_pos[2] += 0.15
q = knife_object.get_quaternion()

obs, reward, done = sawyer_move_to(env, task, target_pos=above_grasp_pos, timeout_s=10.0)
obs, reward, done = sawyer_align_gripper(env, task, approach_direction='down', reference_quat=q, yaw_mode='perpendicular', timeout_s=10.0)
obs, reward, done = sawyer_pick(env, task, target_object=knife_object, target_pos=knife_position, grasp_offset=np.array([0.0, 0.0, -l/4.0]), approach=dict(axis='z', distance=0.15), timeout_s=10.0)

hover_pos = chopping_board_position.copy()
hover_pos[2] += l
obs, reward, done = sawyer_move_to(env, task, target_pos=hover_pos, timeout_s=10.0)
obs, reward, done = sawyer_place(env, task, place_pos=hover_pos, place_offset=np.array([0.0, 0.0, -l]), approach=dict(axis='z', distance=0.15), timeout_s=10.0)

return obs, reward, done