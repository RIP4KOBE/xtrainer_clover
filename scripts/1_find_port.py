from scripts.function_util import log_write, scan_port, free_limit_and_set_one
from dobot_control.dynamixel.driver import DynamixelDriver
from pathlib import Path
from scripts.manipulate_utils import load_ini_data_hands, load_ini_data_gripper
from dobot_control.gripper.dobot_gripper import DobotGripper


if __name__ == "__main__":
    ini_file_path = str(Path(__file__).parent) + "/dobot_config/dobot_settings.ini"
    ini_file, hands_dict = load_ini_data_hands()
    port_list = scan_port()
    assert len(port_list) >= 4, f"At least 4 ports should be detected, but only {len(port_list)} found, please check"

    # find hand port
    for which_hand in hands_dict.keys():
        for _port in port_list:
            try:
                driver = DynamixelDriver(ids=hands_dict[which_hand].joint_ids,
                                         append_id=hands_dict[which_hand].append_id,
                                         port=_port)
                port_list.remove(_port)
                print("Success(hand): ", which_hand, _port)
                ini_file.set(section=which_hand, option="port", value=_port)
                with open(ini_file_path, "w+") as _file:
                    ini_file.write(_file)
                _file.close()
                break
            except Exception as e:
                warnings = e
                continue

    # find gripper port
    print("other port: ", port_list)

    ini_file, gripper_dict = load_ini_data_gripper()
    for which_gripper in gripper_dict.keys():
        for _port in port_list:
            try:
                gripper = DobotGripper(port=_port,
                                       servo_pos=gripper_dict[which_gripper].pos,
                                       id_name=gripper_dict[which_gripper].id_name)
                ini_file.set(section=which_gripper, option="port", value=_port)
                port_list.remove(_port)
                print("Success(gripper): ", which_gripper, _port)

                with open(ini_file_path, "w+") as _file:
                    ini_file.write(_file)
                _file.close()
                break
            except Exception as e:
                warnings = e
                print("***WARNING***: ", warnings)
                continue

    assert not len(port_list), f"Error: find port error ({port_list})"
    print("other port: ", port_list)

