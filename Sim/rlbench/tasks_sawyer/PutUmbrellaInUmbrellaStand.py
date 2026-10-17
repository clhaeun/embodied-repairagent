 umbrella_object = Shape('umbrella')
     stand_object = Shape('stand')
     umbrella_position = np.array(umbrella_object.get_position(), dtype=np.float64)
     stand_position = np.array(stand_object.get_position(), dtype=np.float64)
     umbrella_sizes, _ = get_bbox_sizes(umbrella_object)
     (l, d) = (float(np.max(umbrella_sizes)), float(np.median(umbrella_sizes)))


     handle_pos = umbrella_position.copy()
     handle_pos = handle_pos + np.array([0.0, -0.4 * l, 0.05])
     handle_pos = handle_pos + np.array([-0.05, 0.0, 0.0])


     obs, reward, done = sawyer_move_to(env, task, target_pos=handle_pos, timeout_s=10.0)


     handle_pos = handle_pos + np.array([0.0, 0.0, -0.05])
     obs, reward, done = sawyer_pick(env, task, target_object=umbrella_object, target_pos=handle_pos, grasp_offset=[0.0, 0.0, 0.0])
     hover_pos = stand_position.copy()
     handle_pos = handle_pos + np.array([0.0, 0.0, 0.3])
     obs, reward, done = sawyer_move_to(env, task, target_pos=hover_pos, timeout_s=10.0)


     obs, reward, done = align_to_quaternion(env, task, object_quaternion=umbrella_object.get_quaternion(), yaw_align='parallel', approach_direction='right', tol_rad=1e-3, timeout=10.0)


     hover_pos = hover_pos + np.array([0.0, 0.0, -l])
     obs, reward, done = sawyer_move_to(env, task, target_pos=hover_pos, timeout_s=10.0)

     if done:
         print("[Task] Task successfully completed (done=True).")
         return obs, reward, done
     else:
         print("[Task] Task not completed yet (done=False).")
         return obs, reward, done