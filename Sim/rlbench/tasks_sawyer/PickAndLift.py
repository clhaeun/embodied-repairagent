 target = Shape('pick_and_lift_target')
     success = ProximitySensor('pick_and_lift_success')
     target_pos = target.get_position()
     success_pos = success.get_position()

     # Corrected pick position to ensure proper grasping
     target_pick_pos = target_pos + [0.0, 0.0, 0.05]  # Adjusted z-offset for better grasp

     # Attempt to pick the object
     obs, reward, done = sawyer_pick(env, task, target_object=target, target_pos=target_pick_pos, grasp_offset=np.array([0.0, 0.0, 0.0]))

     if done:
         # Corrected place position to ensure successful placement
         target_move_pos = success_pos + [0.0, 0.0, 0.05]  # Adjusted z-offset for better placement

         # Attempt to place the object
         obs, reward, done = sawyer_place(env, task, place_pos=target_move_pos, grasp_offset=np.array([0.0, 0.0, 0.0]))
     else:
         print('Pick failed, retrying...')

     return obs, reward, done
```
obj['code']
```