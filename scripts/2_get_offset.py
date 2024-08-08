from pathlib import Path
import numpy as np
from dobot_control.dynamixel.driver import DynamixelDriver
MENAGERIE_ROOT: Path = Path(__file__).parent / "third_party" / "mujoco_menagerie"
from scripts.manipulate_utils import load_ini_data_hands
from scripts.function_util import scan_port


def get_config(which_hand, which_hand_config):
    gripper_ids = {"HAND_LEFT": [8], "HAND_RIGHT": [18]}
    driver =DynamixelDriver(ids=which_hand_config.joint_ids+gripper_ids[which_hand],
                            append_id=which_hand_config.append_id,
                            port=which_hand_config.port, baudrate=hands_dict[which_hand].baud_rate)
    # driver.set_torque_mode(False)
    print("--------------------", which_hand, "-------------------")
    pos_joint = driver.get_joints()
    curr_joints = pos_joint[:6]
    print("curr_joints: ", curr_joints)
    print("robot_joints: ", which_hand_config.start_joints)

    dev_pos = [float("%.2f" % (curr_joints[i]-which_hand_config.start_joints[i]*which_hand_config.joint_signs[i]))
               for i in range(6)]
    print("dev(write): ", dev_pos)
    print("dev(*pi/2): ", [(i / np.pi) * 2 for i in curr_joints])
    print("dev(angle): ", [np.rad2deg(i) for i in curr_joints])
    print("----------------------------------------------")
    gripper_on = int(np.rad2deg(pos_joint[-1]) - 0.2)
    gripper_close = int(np.rad2deg(pos_joint[-1]) + 30)
    print(
        "gripper open (degrees)       ",
        gripper_on,
    )
    print(
        "gripper close (degrees)      ",
        gripper_close,
    )
    return dev_pos, [gripper_ids[which_hand][0], gripper_close, gripper_on]


if __name__ == "__main__":
    # left hand nova2
    scan_port()
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file, hands_dict = load_ini_data_hands()
    for _hand in hands_dict.keys():
        offsets, pos_gripper = get_config(_hand, hands_dict[_hand])
        ini_file.set(section=_hand, option="joint_offsets",
                     value=str(offsets).replace("[", '').replace("]", ''))
        ini_file.set(section=_hand, option="gripper_config",
                     value=str(pos_gripper).replace("[", '').replace("]", ''))
        with open(ini_file_path, "w+") as _file:
            ini_file.write(_file)
        _file.close()

