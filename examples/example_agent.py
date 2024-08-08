from dobot_control.agents.dobot_agent import DobotAgent
from scripts.manipulate_utils import load_ini_data_hands


_, hands_dict = load_ini_data_hands()
print("left hand ids: ", hands_dict["HAND_LEFT"].joint_ids)
print("right hand ids: ", hands_dict["HAND_RIGHT"].joint_ids)
left_agent = DobotAgent(which_hand="HAND_LEFT", dobot_config=hands_dict["HAND_LEFT"])
right_agent = DobotAgent(which_hand="HAND_RIGHT", dobot_config=hands_dict["HAND_RIGHT"])

# set torque
right_agent.set_torque(True)
left_agent.set_torque(True)

while 1:
    print(left_agent.act({}), right_agent.act({}))  # joints
    print(left_agent.get_keys(), right_agent.get_keys())  # button status
