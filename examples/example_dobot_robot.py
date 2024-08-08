from dobot_control.robots.dobot import DobotRobot
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light

dobot = DobotRobot("192.168.5.1", no_gripper=False)
set_light(dobot, "green", 1)
while 1:
    print(dobot.get_obs())