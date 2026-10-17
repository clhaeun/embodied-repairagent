 bulb = Shape('bulb')
     bulb_pos = bulb.get_position()
     (obs, reward, done) = sawyer_pick(env, task, target_object=bulb, target_pos=bulb_pos, grasp_offset=[0.0, 0.0, 0.0])
     move_above_pos = bulb_pos + [0.0, 0.0, 0.2]
     (obs, reward, done) = sawyer_move_to(env, task, target_pos=move_above_pos)
     table_pos = [0.5, 0.0, 0.7]
     (obs, reward, done) = sawyer_place(env, task, place_pos=table_pos)
     button = Joint('target_button_joint')
     button_joint_pos = button.get_position()
     button_push_pos = button_joint_pos + [0.0, 0.0, 0.0]
     (obs, reward, done) = sawyer_move_to(env, task, target_pos=button_push_pos)
     (obs, reward, done) = sawyer_pick(env, task, target_object=button, target_pos=button_push_pos, grasp_offset=[0.0, 0.0, 0.0])
     (obs, reward, done) = sawyer_place(env, task, place_pos=button_push_pos)
     (obs, reward, done) = sawyer_close_gripper(env, task, amount=0.0, velocity=0.2)
     bulb = Shape('bulb')
     bulb_pos = bulb.get_position()
     (obs, reward, done) = sawyer_pick(env, task, target_object=bulb, target_pos=bulb_pos, grasp_offset=[0.0, 0.0, 0.0])
     move_above_pos = bulb_pos + [0.0, 0.0, 0.2]
     (obs, reward, done) = sawyer_move_to(env, task, target_pos=move_above_pos)
     table_pos = [0.5, 0.0, 0.7]
     (obs, reward, done) = sawyer_place(env, task, place_pos=table_pos)
     button = Joint('target_button_joint')
     button_joint_pos = button.get_position()
     button_push_pos = button_joint_pos + [0.0, 0.0, 0.0]
     (obs, reward, done