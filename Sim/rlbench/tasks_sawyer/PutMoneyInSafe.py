 dollar_stack_obj = Shape('dollar_stack')
     safe_body_obj = Shape('safe_body')
     dollar_stack_pos = dollar_stack_obj.get_position()
     safe_body_pos = safe_body_obj.get_position()
     (obs, reward, done) = sawyer_move_to(env, task, target_pos=dollar_stack_pos)
     (obs, reward, done) = sawyer_pick(env, task, target_object=dollar_stack_obj, target_pos=dollar_stack_pos, grasp_offset=[0.0, 0.0, 0.0])
     safe_center_pos = safe_body_pos + [0.0, 0.0, 0.1]
     (obs, reward, done) = sawyer_move_to(env, task, target_pos=safe_center_pos)
     safe_entry_pos = safe_body_pos + [0.0, 0.0, 0.05]
     (obs, reward, done) = sawyer_place(env, task, place_pos=safe_entry_pos, place_offset=[0.0, 0.0, 0.0])
     success_detector_pos = safe_body_pos + [0.0, 0.0, 0.2]
     (obs, reward, done) = sawyer_move_to(env, task, target_pos=success_detector_pos)
     if done:
         print('[Task] Task successfully completed (done=True).')
         return (obs, reward, done)
     else:
         print('[Task] Task not completed yet (done=False).')
         return (obs, reward, done)
```
obj['patch']
```