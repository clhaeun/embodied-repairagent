 box_object = Joint('box_joint')
     box_position = box_object.get_position()
     lid_position = np.array(box_position.copy(), dtype=float)
     lid_position[2] += 0.05
     hover_pos = lid_position.copy()
     hover_pos[1] -= 0.3
     hover_pos[2] += 0.1
     obs, reward, done = sawyer_move_to(env, task, target_pos=hover_pos, timeout_s=10.0)
     return obs, reward, done
```
obj['patch']
```