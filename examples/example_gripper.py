from dobot_control.gripper.dobot_gripper import DobotGripper
import time
from scripts.manipulate_utils import load_ini_data_gripper


_, gripper_dict = load_ini_data_gripper()
gripper = DobotGripper(port=gripper_dict["GRIPPER_LEFT"].port,
                       servo_pos=gripper_dict["GRIPPER_LEFT"].pos,
                       id_name=gripper_dict["GRIPPER_LEFT"].id_name)
idx = 10
for i in range(5):
    idx += 20 * i
    gripper.move(idx, 100, 1)
    time.sleep(1)