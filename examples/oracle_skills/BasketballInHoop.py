
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    ball_object = Shape('ball')
    hoop_object = Shape('basket_ball_hoop_respondable')
    ball_position = ball_object.get_position()
    hoop_position = hoop_object.get_position()


    obs, reward, done = pick(env, task, target_pos=ball_position, approach_distance=0.01, approach_axis='z', timeout=10.0)
    

    above_hoop_position = [hoop_position[0], hoop_position[1], hoop_position[2] + 0.15]
    obs, reward, done = place(env, task, target_pos=above_hoop_position, approach_distance=0.1, approach_axis='z', timeout=10.0)

    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")