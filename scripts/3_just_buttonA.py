import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands
from dobot_control.agents.dobot_agent import DobotAgent


@dataclass
class Args:
    agent: str = "dobot"
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    max_step_delta: int = 0.08
    show_img: bool = False
    log_dir: str = "~/bc_data"
    save_data_path = "/home/dobot/projects/test_dir/"


# Thread button: [lock or nor, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))


def button_monitor_realtime(agent):
    # servo
    last_keys_status = np.array(([0, 0], [0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))  # start press
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))

    while 1:
        # time.sleep(0.010)
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status
        # button a
        for i in range(2):
            if keys_press_count[0, 2] % 2 == 0:    # not recording
                if dev_keys[i, 0] == -1:  # button a: start
                    tic = time.time()
                    start_press_status[i, 0] = 1
                if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # button a: end
                    start_press_status[i, 0] = 0
                    toc = time.time()
                    if toc-tic < 0.5:
                        keys_press_count[i, 0] += 1
                        # print(i, keys_press_count[i, 0], "short press", toc-tic)
                        if keys_press_count[i, 0] % 2 == 1:
                            what_to_do[i, 0] = 1
                            # log_write(__file__, "ButtonA: ["+str(i)+"] unlock")
                            print("ButtonA: [" + str(i) + "] unlock", what_to_do)
                        else:
                            what_to_do[i, 0] = 0
                            # log_write(__file__, "ButtonA: [" + str(i) + "] lock")
                            print("ButtonA: [" + str(i) + "] lock", what_to_do)
                    elif toc-tic > 1:
                        keys_press_count[i, 1] += 1
                        # print(i, keys_press_count[i, 1], "long press", toc-tic)
                        if keys_press_count[i, 1] % 2 == 1:
                            what_to_do[i, 1] = 1
                            # log_write(__file__, "ButtonA: [" + str(i) + "] servo")
                            print("ButtonA: [" + str(i) + "] servo")
                        else:
                            what_to_do[i, 1] = 0
                            # log_write(__file__, "ButtonA: [" + str(i) + "] stop servo")
                            print("ButtonA: [" + str(i) + "] stop servo")

        # button B
        if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
            for i in range(2):
                if dev_keys[i, 1] == -1:  # B button pressed
                    start_press_status[i, 1] = 1
                if dev_keys[i, 1] == 1:
                    start_press_status[i, 1] = 0
                    keys_press_count[0, 2] += 1
                    # print(i, keys_press_count[i, 1], "recording")
                    if keys_press_count[0, 2] % 2 == 1:
                        what_to_do[0, 2] = 1
                        # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                    else:
                        what_to_do[0, 2] = 0
                        # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

        last_keys_status = now_keys


def main(args):
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    last_status = np.array(([0, 0, 0], [0, 0, 0]))  # init lock
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent,))
    thread_button.start()
    print("button thread init success...")

    while 1:
        action = agent.act({})
        print(action[0:6], action[7:13])
        print(action[6], action[13])
        dev_what_to_do = what_to_do.copy()-last_status
        last_status = what_to_do.copy()
        # button A: short press event. lock and unlock
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                agent.set_torque(i, not what_to_do[i, 0])
                print(i)


if __name__ == "__main__":
    main(tyro.cli(Args))
