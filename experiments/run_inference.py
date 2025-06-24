import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/ModelTrain/"
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
import torch
# import keyboard
from pynput import keyboard


from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera
from dobot_control.robots.robot_node import ZMQServerRobot
from dobot_control.agents.dp_agent import BimanualDPAgent
from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.robot import BimanualRobot, PrintRobot
from ModelTrain.dp.utils import get_config
from ModelTrain.dp.keypoint_proposer import KeypointProposer
from experiments.run_control import launch_robot_server
from ModelTrain.dp.utils import sixd_to_rotation_vector




from scripts.manipulate_utils import load_ini_data_camera

# from ModelTrain.module.model_module import Imitate_Model
from ModelTrain.dp.train_dp import Agent as DPAgent

@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    agent_name: str = "dp"
    act_ckpt_path: str = "./ckpt/act/tidying_up_bowls_abcefg_mix_0925"
    # dp_ckpt_path: str = "/media/zhuoli/8ECE-77DB/xtrainer/model/DP/dp_plate_wiping_eef_6d_delta_normalization_20250619/last.ckpt"
    dp_ckpt_path: str = "/media/zhuoli/8ECE-77DB/xtrainer/model/DP/dp_plate_wiping_eef_absolute_6d_normalization_20250619/last.ckpt"
    dp_model = None
    act_model = None
    obj_correction = False
    pred_eef_delta = False
    pred_eef_absolute = False
    pred_eef_absolute_6d = True
    pred_eef_delta_6d = False
    bmp_ckpt_pth: str = ("/media/zhuoli/8ECE-77DB/xtrainer/model/BMP/2025.06.24/00.33.33_train_bimanual_motion_prior"
                         "/checkpoints/latest.ckpt")



image_left,image_right,image_top,thread_run=None,None,None,None
lock = threading.Lock()


running = True
mode = "diffusion"

def on_press(key):
    global running, mode
    try:
        if key.char == "1":
            print("Stopping the robot.")
            running = False
        elif key.char == "2":
            print("Modulating trajectory.")
            mode = "modulate"
            running = True
        elif key.char == "3":
            print("Resuming task execution.")
            mode = "diffusion"
            running = True
    except AttributeError:
        pass

def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, thread_run
    if which_cam==0:
        while thread_run:
            image_left, _ = rs_cam.read()
            image_left = image_left[:, :, ::-1]
    elif which_cam==1:
        while thread_run:
            image_right, _ = rs_cam.read()
            image_right = image_right[:, :, ::-1]
    elif which_cam==2:
        while thread_run:
            image_top, _ = rs_cam.read()
            image_top = image_top[:, :, ::-1]
    else:
        print("Camera index error! ")


def main(args):

    # launch the robot server
    dobot_robot_l, dobot_robot_r, dobot_robot = launch_robot_server(args)

    # global variables
    global running, eef_delta, eef_action, eef_action_6d

   # camera init
    global image_left, image_right, image_top, thread_run
    thread_run=True
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

   # get object keypoints if needed
    if args.obj_correction:
       keypoint_config = get_config(config_path="/home/zhuoli/xtrainer_clover/configs/keypoint_config.yaml")
       keypoint_proposer = KeypointProposer(keypoint_config['keypoint_proposer'])
       keypoints = keypoint_proposer.run(visualize_projection=True)

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
        env.step(jnt,np.array([1,1]))
    time.sleep(1)

    # go to the initial photo position
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])  # 用夹爪
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt,np.array([1,1]))

    # Initialize the inference model
    if args.agent_name == "dp":
       # use DP model
       dp_model = BimanualDPAgent(ckpt_path=args.dp_ckpt_path)
       print("DP model init success...")

    else:
        # use ACT model
        # act_model_name = 'policy_best.ckpt'# coaster
        act_model_name = 'policy_last.ckpt'#zip（550）
        # act_model = Imitate_Model(ckpt_dir=args.act_ckpt_path, ckpt_name=act_model_name)
        act_model = Imitate_Model(ckpt_dir='./ckpt/act/tidying_up_bowls_abcefgh_mix_0925', ckpt_name=act_model_name)
        # act_model = Imitate_Model(ckpt_dir='./ckpt/act/dish_washing_20240814', ckpt_name=act_model_name)
        # act_model = Imitate_Model(ckpt_dir='./ckpt/act/pulling_the_zipper_ab_0926', ckpt_name=act_model_name)
        # act_model = Imitate_Model(ckpt_dir='./ckpt/act/tidying_up_coasters_0904', ckpt_name=act_model_name)
        act_model.loadModel()
        print("ACT model init success...")

    episode_len = 700  # The total number of steps to complete the task. Note that it must be less than or equal to parameter 'episode_len' of the corresponding task in file 'ModelTrain.constants'
    t=0
    last_time = 0

    # Initialize the observation
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': []}}
    obs = env.get_obs()
    obs["joint_positions"][6] = 1.0  # Initial position of the gripper
    obs["joint_positions"][13] = 1.0
    observation['qpos'] = obs["joint_positions"]  # Initial value of the joint
    last_action = observation['qpos'].copy()
    last_eef_action = None  # Last eef action, used for DP model modulation

    first = True

    print("The robot begins to perform the task autonomously...")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while t < episode_len:

        if not running:
            time.sleep(0.1)  # Wait when stopped to avoid high CPU usage
            # print("Waiting for the task to resume...")
            continue

        # Obtain the current images
        time0 = time.time()
        # with lock:
        observation['images']['left_wrist'] = image_left
        observation['images']['right_wrist'] = image_right
        observation['images']['top'] = image_top
        if args.show_img:
            imgs = np.hstack((observation['images']['left_wrist'],observation['images']['right_wrist'],observation[
                'images']['top']))
            cv2.imshow("imgs",imgs)
            cv2.waitKey(1)
        time1 = time.time()
        # print("read images time(ms)：",(time1-time0)*1000)

        # Model inference,output joint value (radian) or eef pose value (pos+rotation vector)
        if args.agent_name == "dp":
            dp_observation = {'joint_positions': [], 'left_wrist_rgb': [], 'right_wrist_rgb': [], 'base_rgb': []}
            dp_observation['joint_positions'] = observation['qpos']
            dp_observation['left_wrist_rgb'] = image_left
            dp_observation['right_wrist_rgb'] = image_right
            dp_observation['base_rgb'] = image_top

            if mode == "diffusion":
                prediction, _ = dp_model.act(dp_observation)  # Use planned trajectory

                if args.pred_eef_delta:
                    eef_delta = prediction
                    action = dobot_robot.get_joint_from_eef_delta(eef_delta, obs)
                elif args.pred_eef_absolute:
                    eef_action = prediction
                    joint_state = dobot_robot.get_ik(eef_action)
                    joint_state_l = joint_state[:6]
                    joint_state_r = joint_state[6:12]
                    action=np.concatenate((joint_state_l, [eef_action[6]], joint_state_r, [eef_action[13]]))
                elif args.pred_eef_absolute_6d:
                    # get eef_action_rotvec from eef_action_6d
                    eef_action_6d = prediction
                    eef_action_6d_left_pos = eef_action_6d[:3]
                    eef_action_6d_left_rot = eef_action_6d[3:9]
                    eef_action_6d_left_gripper = eef_action_6d[9]
                    eef_action_6d_right_pos = eef_action_6d[10:13]
                    eef_action_6d_right_rot = eef_action_6d[13:19]
                    eef_action_6d_right_gripper = eef_action_6d[19]

                    left_rotvec = sixd_to_rotation_vector(eef_action_6d_left_rot)
                    right_rotvec = sixd_to_rotation_vector(eef_action_6d_right_rot)

                    eef_action = np.concatenate((eef_action_6d_left_pos, left_rotvec, [eef_action_6d_left_gripper], eef_action_6d_right_pos, right_rotvec, [eef_action_6d_right_gripper]))

                    # compute joint action for safety check
                    joint_state = dobot_robot.get_ik(eef_action)
                    joint_state_l = joint_state[:6]
                    joint_state_r = joint_state[6:12]
                    action = np.concatenate((joint_state_l, [eef_action[6]], joint_state_r, [eef_action[13]]))
                elif args.pred_eef_delta_6d:
                    eef_delta_6d = prediction

                    eef_delta_6d_left_pos = eef_delta_6d[:3]
                    eef_delta_6d_left_rot = eef_delta_6d[3:9]
                    eef_delta_6d_left_gripper = eef_delta_6d[9]
                    eef_delta_6d_right_pos = eef_delta_6d[10:13]
                    eef_delta_6d_right_rot = eef_delta_6d[13:19]
                    eef_delta_6d_right_gripper = eef_delta_6d[19]

                    left_rotvec = sixd_to_rotation_vector(eef_delta_6d_left_rot)
                    right_rotvec = sixd_to_rotation_vector(eef_delta_6d_right_rot)

                    eef_delta_action = np.concatenate(
                        (eef_delta_6d_left_pos, left_rotvec, [eef_delta_6d_left_gripper], eef_delta_6d_right_pos,
                         right_rotvec, [eef_delta_6d_right_gripper]))

                    action = dobot_robot.get_joint_from_eef_delta(eef_delta_action, obs)
                else:
                    action = prediction

            elif mode == "modulate":
                if args.pred_eef_absolute_6d:
                    prediction, modulation_finished = dp_model.act(dp_observation, modulation=True, last_action=last_eef_action)  # Use modulated trajectory
                    eef_action_6d = prediction
                    eef_action_6d_left_pos = eef_action_6d[:3]
                    eef_action_6d_left_rot = eef_action_6d[3:9]
                    eef_action_6d_left_gripper = eef_action_6d[9]
                    eef_action_6d_right_pos = eef_action_6d[10:13]
                    eef_action_6d_right_rot = eef_action_6d[13:19]
                    eef_action_6d_right_gripper = eef_action_6d[19]

                    left_rotvec = sixd_to_rotation_vector(eef_action_6d_left_rot)
                    right_rotvec = sixd_to_rotation_vector(eef_action_6d_right_rot)

                    eef_action = np.concatenate(
                        (eef_action_6d_left_pos, left_rotvec, [eef_action_6d_left_gripper], eef_action_6d_right_pos,
                         right_rotvec, [eef_action_6d_right_gripper]))

                    # compute joint action for safety check
                    joint_state = dobot_robot.get_ik(eef_action)
                    joint_state_l = joint_state[:6]
                    joint_state_r = joint_state[6:12]
                    action = np.concatenate((joint_state_l, [eef_action[6]], joint_state_r, [eef_action[13]]))

        else:
            action = act_model.predict(observation,t)

        # modulated_flag = dp_model.get_modulation_flag()
        # if modulated_flag:
        #     running = False


        # print("infer_action:",action)
        if action[6]>1:
            action[6]=1
        elif action[6]<0:
            action[6] = 0
        if action[13]>1:
            action[13]=1
        elif action[13]<0:
            action[13]=0
        time2 = time.time()
        # print("Model inference time(ms)：", (time2 - time1) * 1000)

        # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
        # [Note]: Modify the protection parameters in this section carefully !

        protect_err = False

        delta = action-last_action
        # print("Joint increment：",delta)
        # if max(delta[0:6])>0.17 or max(delta[7:13])>0.17: # 增量大于10度
        if None: # 增量大于10度

            print("Note!If the joint increment is larger than 10 degrees!!!")
            print("Do you want to continue running? Press the 'Y' key to continue, otherwise press the other button to stop the program!")
            temp_img = np.zeros(shape=(640, 480))
            cv2.imshow("waitKey", temp_img)  # Make cv2.waitkey(0) work
            key = cv2.waitKey(0)
            # Check the keys
            if key == ord('y') or key == ord('Y') :  # If press the 'Y' key
                cv2.destroyWindow("waitKey")
                # go to the position slowly
                max_delta = (np.abs(last_action - action)).max()
                steps = min(int(max_delta / 0.001), 100)
                for jnt in np.linspace(last_action, action, steps):
                    env.step(jnt,np.array([1,1]))
                first = False
            else:
                protect_err = True
                cv2.destroyAllWindows()

        # Left arm joint angle limitations:  -150<J3<0    J4>-35  (Note: This angle needs to be converted to radians)
        # right arm joint angle limitations:  150>J3>0    J4<35   (Note: This angle needs to be converted to radians)
        # if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and \
        #         (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):
        #     print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
        #     print(action)
        #     protect_err = True

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
        # print("get pos time(ms):", (t2 - t1)* 1000)

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
                env.step(jnt,np.array([1,1]))
            first = False

        last_action = action.copy()
        last_eef_action = eef_action_6d.copy() if 'eef_action_6d' in locals() else None

        # Control robot movement
        time3 = time.time()
        if args.pred_eef_delta:
            eef_action = dobot_robot.get_eef_action(eef_delta, obs["ee_pos_quat"])
            # obs = env.step_eef(eef_action, np.array([1, 1]))
            obs = env.step(action, np.array([1, 1]))
        elif args.pred_eef_absolute or args.pred_eef_absolute_6d or args.pred_eef_delta_6d:
            # obs = env.step_eef(eef_action, np.array([1, 1]))
            obs = env.step(action, np.array([1, 1]))
        else:
            obs = env.step(action, np.array([1, 1]))

        time4 = time.time()

        # Obtain the current joint value of the robots (including the gripper)
        obs["joint_positions"][6] = action[6]   # In order to decrease acquisition time, the last action of the gripper is taken as its current observation
        obs["joint_positions"][13] = action[13]
        observation['qpos'] = obs["joint_positions"]

        # print("Read joint value time(ms)：", (time4 - time3) * 1000)
        t +=1
        # print("The total time(ms):", (time4 - time0) * 1000)


        if args.agent_name == "dp" and mode == "modulate" and modulation_finished:
            print("Trajectory execution finished. Waiting for next user input...")
            running = False


    thread_run = False
    listener.stop()
    print("Task accomplished")

    # Return to the starting position
    # ...


if __name__ == "__main__":
    main(tyro.cli(Args))

