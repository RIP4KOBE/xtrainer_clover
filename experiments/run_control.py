import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame, save_dp_frame
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
import datetime
from pathlib import Path

@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = False
    save_data_path = "/home/zhuoli/pan_new/datasets/"
    project_name = "dp_test_20240820"
    # whether to collect data for diffusion policy
    dp_data = True
    dp_save_png = False
    dp_dir_created = False


# Thread button: [lock or nor, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])
# dp_time = np.array([20240507161455])
using_sensor = False
is_falling = np.array([0])

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
        # more than one start servo
        for i in range(2):
            if dev_keys[i, 1] == -1:  # B button pressed
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                        # new recording
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        # dp_time[0] = int(now_time.strftime("%m%d_%H%M%S"))
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1
                    # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

        last_keys_status = now_keys


# Thread: camera
npy_list = np.array([np.zeros(480*640*3), np.zeros(480*640*3), np.zeros(480*640*3)])
npy_len_list = np.array([0, 0, 0])
img_list = np.array([np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3))])


def run_thread_cam(rs_cam, which_cam):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while 1:
        image_cam, _ = rs_cam.read()
        image_cam = image_cam[:, :, ::-1]
        img_list[which_cam] = image_cam
        _, image_ = cv2.imencode('.jpg', image_cam, encode_param)
        npy_list[which_cam][:len(image_)] = image_
        npy_len_list[which_cam] = len(image_)



def dh_transformation_matrix(theta, d, a, alpha):
    """
    Create the DH transformation matrix
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1]
    ])

def claw_width(coef):
    """
    Calculate the claw width
    """
    claw_servo = 2.3818 - coef * 1.5401
    cos_claw_servo = np.cos(claw_servo)
    claw_wid = 0.03 * cos_claw_servo + 0.5 * np.sqrt(0.0036 * cos_claw_servo ** 2 + 0.0028)
    return claw_wid

def forward_kinematics(q0, q1, q2, q3, q4, q5, y):
    """
    Compute the forward kinematics
    """
    dh_params = [
        (q0, 0.2234, 0, np.pi / 2),
        (q1 - np.pi / 2, 0, -0.280, 0),
        (q2, 0, -0.225, 0),
        (q3 - np.pi / 2, 0.1175, 0, np.pi / 2),
        (q4, 0.120, 0, -np.pi / 2),
        (q5, 0.088, 0, 0)
    ]

    t = np.eye(4)
    for params in dh_params:
        t = np.dot(t, dh_transformation_matrix(*params))
    t_tool = np.eye(4)
    t_tool[:3, 3] = np.array([0, y, 0.2])
    t_final = np.dot(t, t_tool)
    pos = t_final[:3, 3]
    return pos


def calculate_vel_pos(action, last_action, total_time):
    """
    Calculate the velocity for forward kinematics
    """
    claw_left = claw_width(action[6])
    claw_right = claw_width(action[13])

    positions = {}
    vel = {}

    for side in ['left', 'right']:
        for paw in ['left', 'right']:
            coef = 1 if paw == 'left' else -1
            claw = claw_left if side == 'left' else claw_right
            claw *= coef

            current_fk = forward_kinematics(*action[0:6] if side == 'left' else action[7:13], claw)
            last_fk = forward_kinematics(*last_action[0:6] if side == 'left' else last_action[7:13], claw)

            positions[f'{side}_{paw}'] = current_fk
            vel[f'{side}_{paw}'] = (current_fk - last_fk) / total_time

    return positions, vel

# Check that the positions is within a safe zone
def is_within_safe_position(position, x_range, y_range, z_min):
    return x_range[0] <= position[0] <= x_range[1] and \
           y_range[0] <= position[1] <= y_range[1] and \
           position[2] > z_min


def check_pose_protection(positions, vel, what_to_do):
    protect_err = False
    warnings = []

    delta_left_left = vel['left_left']
    delta_left_right = vel['left_right']
    delta_right_left = vel['right_left']
    delta_right_right = vel['right_right']

    positions_mm = {key: value * 1000 for key, value in positions.items()}
    # Define a safe zone
    # left arm (jaw tip position) limit:  290>x>-450  -750<Y<-160  z>44;
    # right arm (jaw tip position) limit:  450>x>-290  -750<Y<-160  z>42;
    x_range_left = (-450, 290)
    x_range_right = (-290, 450)
    y_range = (-750, -160)
    z_range_left = 44
    z_range_right = 42

    if what_to_do[0, 1]:  # The left hand is in sync
        # Z direction speed limit -1 m/s
        if delta_left_left[2] < -1 or delta_left_right[2] < -1:
            warnings.append("[Warn]:The left robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_left_left: {delta_left_left[2]}")
            protect_err = True
        # Left arm working space limitation
        positions_to_check = ['left_left', 'left_right']
        x_ranges = [x_range_left, x_range_left]
        z_ranges = [z_range_left, z_range_left]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The left arm is out of the safe zone!")
            protect_err = True

    if what_to_do[1, 1]:  # The right hand is in sync
        # Z direction speed limit -1 m/s
        if delta_right_left[2] < -1 or delta_right_right[2] < -1:
            warnings.append("[Warn]:The right robot speed of the TCP is moving too fast!")
            warnings.append(f"delta_right_left: {delta_right_left[2]}")
            protect_err = True
        # Right arm working space limitation
        positions_to_check = ['right_left', 'right_right']
        x_ranges = [x_range_right, x_range_right]
        z_ranges = [z_range_right, z_range_right]
        if not all(is_within_safe_position(positions_mm[pos], x_range, y_range, z_range)
                   for pos, x_range, z_range in zip(positions_to_check, x_ranges, z_ranges)):
            warnings.append("[Warn]:The right arm is out of the safe zone!")
            protect_err = True

    for warning in warnings:
        print(warning)

    return protect_err

def check_joint_safety(action):
    protect_err = False
    if not (action[2] < 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    if not (action[9] > 0):
        print("[Warn]:The J3 joints of the robotic arm are out of the safe position! ")
        print(action)
        protect_err = True
    return protect_err


def main(args):
    # create dataset file path
    save_dir = args.save_data_path+args.project_name+"/collect_data"
    mk_dir(save_dir)

    # camera init
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    # agent init
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    # pose init
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

    # button status init
    last_status = np.array(([0, 0, 0], [0, 0, 0]))  # init lock
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent, ))
    thread_button.start()
    print("button thread init success...")


    print("-------------------------Ok, let's start------------------------")
    idx = 0
    safe_limit = 0
    total_time = 0.04

    while 1:
        tic = time.time()

        assert thread_cam_top.is_alive(), "Error: please check the top camera!"
        assert thread_cam_left.is_alive(), "Error: please check the left camera!"
        assert thread_cam_right.is_alive(), "Error: please check the right camera!"
        assert not is_falling, "sensor   detection!"

        action = agent.act({})
        print(action)
        dev_what_to_do = what_to_do.copy()-last_status
        last_status = what_to_do.copy()
        # button A: short press event. lock and unlock
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                agent.set_torque(i, not what_to_do[i, 0])

        # button A: long press event. servo or not
        if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
            # pose check between main hand and the follower
            print("dynamic approach")
            for i in range(2):
                if what_to_do[i, 1]:
                    agent.set_torque(i, True)
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            last_action = dynamic_approach(env, agent, flag_in)
            for i in range(2):
                if what_to_do[i, 0]:
                    if what_to_do[i, 1]:
                        agent.set_torque(i, False)
            start_servo = True
            obs = env.get_obs()
            if curr_light != "green":
                curr_light = set_light(env, "yellow", 1)

        if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            if what_to_do[0, 1] == 0 and what_to_do[1, 1] == 0:
                set_light(env, "green", 0)

        if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
            action = agent.act({})
            err3, action = servo_action_check(action, last_action, flag_in)
            assert err3 != 0, set_light(env, "red", 1)

            # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
            # [Note]: Modify the protection parameters in this section carefully !
            protect_err = [False, False]
            if (safe_limit < 1):
                safe_limit = safe_limit + 1
            else:
                positions, vel = calculate_vel_pos(action, last_action, total_time)
                protect_err[0] = check_pose_protection(positions, vel, what_to_do)
                protect_err[1] = check_joint_safety(action)
            if any(protect_err):
                set_light(env, "red", 1)
                time.sleep(1)
                exit()
            # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××

            # button B: recording or not
            if dev_what_to_do[0, 2] == 1:
                curr_light = set_light(env, "green", 1)
            elif dev_what_to_do[0, 2] == -1:
                curr_light = set_light(env, "yellow", 1)
            if what_to_do[0, 2] == 1:
                
                # collect data for Diffusion Policy
                if args.dp_data:
                    dt = datetime.datetime.now()
                    dp_save_dir = save_dir + f"/{dt_time[0]}/"
                    mk_dir(dp_save_dir)

                    # save img data to obs
                    obs["base_rgb"] = img_list[0].astype(np.uint8)
                    obs["left_wrist_rgb"] = img_list[1].astype(np.uint8)
                    obs["right_wrist_rgb"] = img_list[2].astype(np.uint8)

                    save_dp_frame(
                        dp_save_dir,
                        dt,
                        obs,
                        action,
                        save_png=args.dp_save_png,
                    )

                # collect data for ACT
                else:
                    idx += 1
                    left_dir = save_dir + f"/{dt_time[0]}/leftImg/"
                    right_dir = save_dir + f"/{dt_time[0]}/rightImg/"
                    top_dir = save_dir + f"/{dt_time[0]}/topImg/"
                    mk_dir(right_dir)
                    mk_dir(top_dir)
                    if mk_dir(left_dir):
                        idx = 0

                    obs_dir = save_dir + f"/{dt_time[0]}/observation/"
                    mk_dir(obs_dir)

                    cv2.imwrite(top_dir + f"{idx}.jpg", img_list[0])
                    cv2.imwrite(left_dir + f"{idx}.jpg", img_list[1])
                    cv2.imwrite(right_dir + f"{idx}.jpg", img_list[2])
                    save_frame(obs_dir, idx, obs, action)

            obs = env.step(action, flag_in)
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            last_action = action
        else:
            start_servo = False
            safe_limit = 0

        # img show
        if args.show_img:
            show_canvas[:, :640] = np.asarray(img_list[0], dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(img_list[1], dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2], dtype="uint8")
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        toc = time.time()
        total_time = toc-tic
        print("total time: ", total_time)


if __name__ == "__main__":
    main(tyro.cli(Args))
