```python
def run_skill(env, task, descriptions=None, obs=None, variation_index: int = 1):
    rubbish = Shape('rubbish')
    bin_sensor = ProximitySensor('success')

    rubbish_pos = rubbish.get_position()
    bin_pos = bin_sensor.get_position()


    q = rubbish.get_quaternion()
    obs, reward, done = ur5_align_gripper(env, task, reference_quat=q, yaw_mode='parallel', approach_direction='front', timeout_s=10.0)
    

    obs, reward, done = ur5_grasp_at(env, task, grasp_pos=rubbish_pos, approach={'axis':'z', 'distance':0.1}, timeout_s=10.0)


    above_bin_pos = bin_pos + [0.0, 0.0, 0.2]
    obs, reward, done = ur5_move_to(env, task, target_pos=above_bin_pos, timeout_s=10.0)


    obs, reward, done = ur5_release_at(env, task, place_pos=bin_pos, approach={'axis':'z', 'distance':0.1}, timeout_s=10.0)

    return obs, reward, done
```