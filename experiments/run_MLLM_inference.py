import sys
import os

from Dp_action_calling import get_MLLM_response_with_audio_demo

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ModelTrain/"
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
import torch
import json
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.agents.dp_agent import BimanualDPAgent

from utils.Dp_action_calling import get_curi_response_with_audio, process_robot_actions
from scripts.manipulate_utils import load_ini_data_camera

from ModelTrain.module.model_module import Imitate_Model
from ModelTrain.dp.pipeline import Agent as DPAgent


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    agent_name: str = "act"
    act_ckpt_path: str = "./ckpt/act/tidying_up_bowls_abcefg_mix_0925"
    dp_ckpt_path: str = "/home/zhuoli/xtrainer_clover/ModelTrain/ckpt/dp/dp_tidying_up_bowls_a_0920_3cam/last.ckpt"
    dp_model = None
    act_model = None


image_left, image_right, image_top, thread_run = None, None, None, None
lock = threading.Lock()


def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, thread_run
    if which_cam == 0:
        while thread_run:
            image_left, _ = rs_cam.read()
            image_left = image_left[:, :, ::-1]
    elif which_cam == 1:
        while thread_run:
            image_right, _ = rs_cam.read()
            image_right = image_right[:, :, ::-1]
    elif which_cam == 2:
        while thread_run:
            image_top, _ = rs_cam.read()
            image_top = image_top[:, :, ::-1]
    else:
        print("Camera index error! ")


def main(args):
    # camera init
    global image_left, image_right, image_top, thread_run
    thread_run = True
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs2 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_top.start()
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    # robot init
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    print("robot init success...")

    # go to the safe position
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))
    time.sleep(1)

    # go to the initial photo position
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])  # 用夹爪
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))

    # Call the MLM to determine the low_level action
    with open('/home/zhuoli/xtrainer_clover/curigpt_ros/config/config.json', 'r') as config_file:
        config = json.load(config_file)
        # accessing configuration variables
    api_key = config['openai_api_key']
    base_url = config['base_url']
    user_input_filename = config['user_input_filename_demo']
    curigpt_output_filename = config['curigpt_output_filename_demo_bowls']
    depth_img_path = config['depth_img_path']
    rgb_img_path = config['rgb_img_path']
    local_img_path1 = config['local_img_path1']
    local_img_path2 = config['local_img_path2']
    model_name = config['model_name']

    base_multimodal_prompt = [
        {
            "role": "system",
            "content": [{
                "text": '''You are an excellent responser of human instructions for household tabletop tasks. Given an verbal instruction and an image, you respond to the human instruction and select appropriate robot actions if necessary. Note that you only need to focus on the partial area of the table top covered by a light green tablecloth in the image.

                              Your response must be output in a structured JSON format and contain the following two keywords:
                              - "robot_response" for the verbal response to the human instruction.
                              - "robot_actions" for the description of the physical action you will perform, including the specific action name.
                              Must add the "," delimiter between the two keywords to ensure proper JSON formatting.

                              Two robot actions are available to you:
                              - tidying_up_bowl: The robot first grasps the spoon and places it into the bowl, then picks up the bowl's lid and puts it on the bowl.
                              - tidying_up_coaster: The robot picks up the coaster and places it into its black case.
                              Note that the choice of action should be made by yourself according to the objects in the image you see. You need to choose the most reasonable action.'''
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path1},
                {"text": "hey, what do you see right now?"},
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Now I see a table whose top is covered by a light green tablecloth, and a bowl is in the center of the table. There is also a spoon and a lid left and right to the bowl respectively.",
                    "robot_actions": None
                }, indent=4)
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path1},
                {"text": "Can you help me tidy up the table?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Sure, I will first pick the spoon, then place it in the bowl, and finally pick and put the lid on the bowl.",
                    "robot_actions": [
                        {
                            "action": "tidying_up_bowl"
                        }
                    ]
                }, indent=4)
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path2},
                {"text": "hey, what do you see right now?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Now I see a table whose top is covered by a light green tablecloth, a blue coaster placed on the table, and a coaster's black case in front of the coaster.",
                    "robot_actions": None
                }, indent=4)
            }]
        },
        {
            "role": "user",
            "content": [
                {"image": local_img_path2},
                {"text": "Can you help me tidy up the table?"}
            ]
        },
        {
            "role": "assistant",
            "content": [{
                "text": json.dumps({
                    "robot_response": "Sure, I will pick the coaster and then place it into its case.",
                    "robot_actions": [
                        {
                            "action": "tidying_up_coaster"
                        }
                    ]
                }, indent=4)
            }]
        }
    ]

    model_path = get_MLLM_response_with_audio_demo(model_name, api_key, base_url, user_input_filename,
                                              curigpt_output_filename)

    # Initialize the inference model
    if args.agent_name == "dp":
        # use DP model
        dp_model = BimanualDPAgent(ckpt_path=args.dp_ckpt_path)
        print("DP model init success...")

    else:
        # use ACT model
        act_model_name = 'policy_last.ckpt'  # zip（550）
        act_model = Imitate_Model(ckpt_dir=model_path, ckpt_name=act_model_name)
        act_model.loadModel()
        print("ACT model init success...")

    episode_len = 500  # The total number of steps to complete the task. Note that it must be less than or equal to parameter 'episode_len' of the corresponding task in file 'ModelTrain.constants'
    t = 0
    last_time = 0

    # Initialize the observation
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': []}}
    obs = env.get_obs()
    obs["joint_positions"][6] = 1.0  # Initial position of the gripper
    obs["joint_positions"][13] = 1.0
    observation['qpos'] = obs["joint_positions"]  # Initial value of the joint
    last_action = observation['qpos'].copy()

    first = True

    print("The robot begins to perform tasks autonomously...")
    while t < episode_len:
        # Obtain the current images
        time0 = time.time()
        # with lock:
        observation['images']['left_wrist'] = image_left
        observation['images']['right_wrist'] = image_right
        observation['images']['top'] = image_top
        if args.show_img:
            imgs = np.hstack((observation['images']['left_wrist'], observation['images']['right_wrist'],
                              observation['images']['top']))
            cv2.imshow("imgs", imgs)
            cv2.waitKey(1)
        time1 = time.time()
        print("read images time(ms)：", (time1 - time0) * 1000)

        # Model inference,output joint value (radian)
        if args.agent_name == "dp":
            dp_observation = {'joint_positions': [], 'left_wrist_rgb': [], 'right_wrist_rgb': [], 'base_rgb': []}
            dp_observation['joint_positions'] = observation['qpos']
            dp_observation['left_wrist_rgb'] = image_left
            dp_observation['right_wrist_rgb'] = image_right
            dp_observation['base_rgb'] = image_top
            action = dp_model.act(dp_observation)

        else:
            action = act_model.predict(observation, t)

        # print("infer_action:",action)
        if action[6] > 1:
            action[6] = 1
        elif action[6] < 0:
            action[6] = 0
        if action[13] > 1:
            action[13] = 1
        elif action[13] < 0:
            action[13] = 0
        time2 = time.time()
        print("Model inference time(ms)：", (time2 - time1) * 1000)

        # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
        # [Note]: Modify the protection parameters in this section carefully !

        protect_err = False

        delta = action - last_action
        print("Joint increment：", delta)
        if max(delta[0:6]) > 0.17 or max(delta[7:13]) > 0.17:  # 增量大于10度
            print("Note!If the joint increment is larger than 10 degrees!!!")
            print(
                "Do you want to continue running? Press the 'Y' key to continue, otherwise press the other button to stop the program!")
            temp_img = np.zeros(shape=(640, 480))
            cv2.imshow("waitKey", temp_img)  # Make cv2.waitkey(0) work
            key = cv2.waitKey(0)
            # Check the keys
            if key == ord('y') or key == ord('Y'):  # If press the 'Y' key
                cv2.destroyWindow("waitKey")
                # go to the position slowly
                max_delta = (np.abs(last_action - action)).max()
                steps = min(int(max_delta / 0.001), 100)
                for jnt in np.linspace(last_action, action, steps):
                    env.step(jnt, np.array([1, 1]))
                first = False
            else:
                protect_err = True
                cv2.destroyAllWindows()

        # Left arm joint angle limitations:  -150<J3<0    J4>-35  (Note: This angle needs to be converted to radians)
        # right arm joint angle limitations:  150>J3>0    J4<35   (Note: This angle needs to be converted to radians)
        if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and \
                (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):
            print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
            print(action)
            protect_err = True

        # left arm (jaw tip position) limit:  210>x>-410  -700<Y<-210  z>47;
        # right arm (jaw tip position) limit:  410>x>-210  -700<Y<-210  z>47;
        t1 = time.time()
        pos = env.get_XYZrxryrz_state()
        if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 42) and \
                (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 42)):
            print("[Warn]:The robot arm XYZ is out of the safe position! ")
            print(pos)
            protect_err = True
        t2 = time.time()
        print("get pos time(ms):", (t2 - t1) * 1000)

        if protect_err:
            env.set_do_status([3, 0])  # yellow light off
            env.set_do_status([2, 0])  # green light off
            env.set_do_status([1, 1])  # red light on
            time.sleep(1)
            exit()
        # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

        if first:
            max_delta = (np.abs(last_action - action)).max()
            steps = min(int(max_delta / 0.001), 100)
            for jnt in np.linspace(last_action, action, steps):
                env.step(jnt, np.array([1, 1]))
            first = False

        last_action = action.copy()

        # Control robot movement
        time3 = time.time()
        obs = env.step(action, np.array([1, 1]))
        time4 = time.time()

        # Obtain the current joint value of the robots (including the gripper)
        obs["joint_positions"][6] = action[
            6]  # In order to decrease acquisition time, the last action of the gripper is taken as its current observation
        obs["joint_positions"][13] = action[13]
        observation['qpos'] = obs["joint_positions"]

        print("Read joint value time(ms)：", (time4 - time3) * 1000)
        t += 1
        print("The total time(ms):", (time4 - time0) * 1000)

    thread_run = False
    print("Task accomplished")

    # Return to the starting position
    # ...


if __name__ == "__main__":
    main(tyro.cli(Args))

