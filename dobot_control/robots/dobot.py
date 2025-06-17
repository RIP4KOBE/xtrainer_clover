from typing import Dict
import numpy as np
import time
import quaternion
from dobot_control.robots.robot import Robot
import struct
import sys
from scripts.manipulate_utils import load_ini_data_hands, load_ini_data_gripper
from scipy.spatial.transform import Rotation as R
from scripts.manipulate_utils import forward_kinematics, claw_width
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, \
    servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from threading import Event, Lock, Thread


class DobotRobot(Robot):
    """A class representing a UR robot."""

    def __init__(self, robot_ip: str = "192.168.5.1", no_gripper: bool = False, robot_number: int = 1):
        from dobot_control.robots import dobot_api
        self.robot_number = robot_number
        self.frequency_ = 1 / 0.015
        # Set delta time to be used by receiveCallback
        self.delta_time_ = 1 / self.frequency_
        [print("in dobot robot") for _ in range(4)]
        self.robot = dobot_api.DobotApiMove(robot_ip, 30003)  # 运动指令的端�?
        # try:
        #     self.robot = dobot_api.DobotApiMove(robot_ip, 30003)  # 运动指令的端�?
        # except Exception as e:
        #     print(e)
        #     print("Please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
        #     sys.exit()
        self.robot_ip = robot_ip
        self.r_inter = dobot_api.DobotApiDashboard(robot_ip, 29999)  # 获取信息指令的端�?
        self.r_inter.EnableRobot()  # 上使能机械臂，发送指令前必须执行�?
        self.r_inter.SpeedFactor(20)  # 全局速度设置
        self.r_inter.AccJ(20)  # 全局加速度设置
        self.r_inter.SpeedJ(20)  # 全局速度设置
        self.r_inter.SetTool(1, 0, 0, 197, 0, 0, 0)  # 设置夹爪工具坐标系
        self.r_inter.Tool(1)  # 设置当前工具坐标系，使得Getpose()获取坐标时为此工具下

        self.com_list = {"192.168.5.1": "GRIPPER_LEFT", "192.168.5.2": "GRIPPER_RIGHT"}  # left, right
        _, gripper_dict = load_ini_data_gripper()
        self.gripper_list = gripper_dict[self.com_list[robot_ip]].pos
        self.gripper_id_name = gripper_dict[self.com_list[robot_ip]].id_name

        if not no_gripper:
            from dobot_control.gripper.dobot_gripper import DobotGripper
            self.gripper = DobotGripper(port=gripper_dict[self.com_list[robot_ip]].port,
                                        id_name=self.gripper_id_name,
                                        servo_pos=self.gripper_list)
            print("gripper connected")
            self.gripper.move(0, 100, 1)
            time.sleep(0.3)
            self.gripper.move(255, 100, 1)

        self._free_drive = False  # 拖拽模式
        self.r_inter.StopDrag()  # 关闭拖拽
        self._use_gripper = not no_gripper

        self.robot_status = dobot_api.DobotApiStatus(robot_ip, 30004)
        self._stop_thread = Event()
        self._start_reading_thread()
        self._lock = Lock()
        self.robot_is_err = False

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self.get_robot_err)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def get_robot_err(self):
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            with self._lock:
                if self.robot_status.get_error():
                    self.robot_is_err = True
                    assert not self.robot_is_err, f"{self.robot_ip}: error!"

    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        if self._use_gripper:
            return 7
        return 7

    def _get_gripper_pos(self) -> float:
        # import time
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        gripper_pos = self.gripper.get_current_position()
        # gripper_pos = self.get_current_position()
        # print("gripper_pos:",gripper_pos)
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return np.abs(1 - gripper_pos / 255)
        # return np.abs(gripper_pos / 255)

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        # log_write(str(self.robot_ip) + ": getpose_start")
        robot_joints_angle = list(map(float, self.r_inter.GetAngle().split("{")[1].split("}")[0].split(",")))  # 单位：度�?
        robot_joints = [np.deg2rad(robot_joint) for robot_joint in robot_joints_angle]  # 单位：弧�?
        if self._use_gripper:
            gripper_pos = [1.0]
            pos = np.append(robot_joints, gripper_pos)
        else:
            gripper_pos = [1.0]
            pos = np.append(robot_joints, gripper_pos)
        return pos

    def get_eef_pose(self) -> np.ndarray:
        """Get the current pose of the leader robot's end effector.

        Returns:
            T: The current pose of the leader robot's end effector.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        pos_rot = np.array(list(map(float, self.r_inter.GetPose().split("{")[1].split("}")[0].split(",")))) #
        # 单位：位置:mm, 姿态:度数
        # print("get_eef_pose:", pos_rot)
        pos = pos_rot[:3] / 1000
        rot = R.from_euler('xyz', pos_rot[3:6], degrees=True).as_rotvec() # Convert zyx Euler angles to rotation vector
        pos_rot = np.concatenate((pos, rot))  # [x, y, z, rx, ry, rz]

        return pos_rot

    def get_fk(self, joint_state) -> np.ndarray:
        """Get the current forward kinematics state of the robot.

        Returns:
            T: The current forward kinematics state of the robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        joint_state = np.rad2deg(joint_state)

        fk_sol = np.array(list(map(float, self.r_inter.PositiveSolution(
            joint_state[0], joint_state[1], joint_state[2],
            joint_state[3], joint_state[4], joint_state[5]
        ,0,1).split("{")[1].split("}")[0].split(","))))
        pos = fk_sol[:3] / 1000
        rot = R.from_euler('xyz', fk_sol[3:6], degrees=True).as_rotvec() # Convert zyx Euler angles to rotation vector
        fk_sol = np.concatenate((pos, rot))  # [x, y, z, rx, ry, rz]

        return fk_sol

    def get_ik(self, eef_state: np.ndarray) -> np.ndarray:
        """Get the current inverse kinematics state of the robot.

        Args:
            eef_state (np.ndarray): The end effector state to get the inverse kinematics for.

        Returns:
            T: The current inverse kinematics state of the robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        pos = eef_state[:3] * 1000
        rot = R.from_rotvec(eef_state[3:6]).as_euler('xyz', degrees=True)
        ik_sol = np.array(list(map(float, self.r_inter.InverseSolution(
            pos[0], pos[1], pos[2],
            rot[0], rot[1], rot[2], 0, 1).split("{")[1].split("}")[0].split(","))))
        joint_state = np.deg2rad(ik_sol[:6])  # Convert degrees to radians
        return joint_state


    def get_eef_action(self, eef_delta: np.ndarray, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get the end effector action of the robot from the current observation.

        Args:
            eef_delta (np.ndarray): The eef_delta action to get the end effector action for.
            obs (Dict[str, np.ndarray]): The current observation of the robot.

        Returns:
            np.ndarray: The end effector action of the robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        assert "ee_pos_quat" in obs, "Observation must contain 'ee_pos_quat' key"
        current_eef_pose = obs["ee_pos_quat"]
        pos_delta = eef_delta[:3]
        rot_delta = eef_delta[3:6]
        pos = pos_delta[:3] + current_eef_pose[:3]
        # quaternion multiplication
        rot = quaternion.as_rotation_vector(
            quaternion.from_rotation_vector(rot_delta)
            * quaternion.from_rotation_vector(current_eef_pose[3:])
        )
        pos_rot = np.concatenate((pos, rot))  # [x, y, z, rx, ry, rz]
        eef_action = np.concatenate((pos_rot, np.array(eef_delta[-1]))) # [x, y, z, rx, ry, rz, gripper_pos]

        return eef_action

    def get_joint_from_eef_delta(self, eef_delta: np.ndarray, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Get the current joint action of the robot from the eef_delta action.

        Args:
            eef_delta (np.ndarray): The eef_delta action to get the joint action for.
            obs (Dict[str, np.ndarray]): The current observation of the robot.

        Returns:
            T: The current joint action of the robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        assert len(eef_delta) == 7, "eef_action must be of length 7 (6 for pose and 1 for gripper)"
        # current_eef_pose = obs["ee_pos_quat"] # [x, y, z, rx, ry, rz]
        # pos_delta = eef_delta[:3]
        # rot_delta = eef_delta[3:6]
        # pos = pos_delta[:3] + current_eef_pose[:3]
        # # quaternion multiplication
        # rot = quaternion.as_rotation_vector(
        #     quaternion.from_rotation_vector(rot_delta)
        #     * quaternion.from_rotation_vector(current_eef_pose[3:])
        # )
        # pos_rot = np.concatenate((pos, rot))  # [x, y, z, rx, ry, rz]
        eef_action = self.get_eef_action(eef_delta, obs)  # [x, y, z, rx, ry, rz, gripper_pos]
        ik_sol = self.get_ik(eef_action[:6]) # Get the inverse kinematics solution for the eef action
        joint_action = np.concatenate((ik_sol, np.array(eef_action[-1])))

        assert len(joint_action) == 7, "joint_action must be of length 7 (6 for joints and 1 for gripper)"

        return joint_action

    def get_XYZrxryrz_state(self) -> np.ndarray:
        """Get the current X Y Z rx ry rz state of the robot.
        Returns:
            T: The current X Y Z rx ry rz state of the robot.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        pos = list(map(float, self.r_inter.GetPose().split("{")[1].split("}")[0].split(",")))  # 单位：度数
        return pos


    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        robot_joints_angle = joint_state[:6]  # 单位：弧�?
        robot_joints = [np.rad2deg(robot_joint) for robot_joint in robot_joints_angle]
        self.robot.ServoJ(robot_joints[0],
                          robot_joints[1],
                          robot_joints[2],
                          robot_joints[3],
                          robot_joints[4],
                          robot_joints[5],
                          0.03)
        if self._use_gripper:
            gripper_pos = int(joint_state[-1] * 255)
            self.gripper.move(gripper_pos, 100, 1)
        return 1


    def command_eef_state(self, eef_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            pose_state (np.ndarray): The state to command the leader robot to.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"

        pos_rot = eef_state[:6] # 单位：位置:m, 姿态:rotation vevtor radians
        # print("dobot single arm eef state:", pos_rot)
        pos = pos_rot[:3] * 1000
        rot = R.from_rotvec(pos_rot[3:6]).as_euler('xyz', degrees=True)
        pos_rot = np.concatenate((pos, rot))  # [x, y, z, rx, ry, rz]
        # print("start command_eef_state:", pos_rot)

        self.robot.ServoP(pos_rot[0],pos_rot[1], pos_rot[2],
                        pos_rot[3], pos_rot[4], pos_rot[5])

        if self._use_gripper:
            gripper_pos = int(eef_state[-1] * 255)
            self.gripper.move(gripper_pos, 100, 1)

        return 1

    def moveJ(self, joint_state: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        robot_joints_angle = joint_state[:6]  # ��λ������
        robot_joints = [np.rad2deg(robot_joint) for robot_joint in robot_joints_angle]  # ��λ���Ƕ�
        tic = time.time()
        print(robot_joints)
        self.robot.JointMovJ(robot_joints[0],
                             robot_joints[1],
                             robot_joints[2],
                             robot_joints[3],
                             robot_joints[4],
                             robot_joints[5])
        toc = time.time()
        if self._use_gripper:
            tic = time.time()
            gripper_pos = int(joint_state[-1] * 255)
            # print("gripper_pos:", gripper_pos)
            self.gripper.move(gripper_pos, 100, 1)
            # log_write(str(self.robot_ip) + ": gripper_move")
            toc = time.time()

    def freedrive_enabled(self) -> bool:
        """Check if the robot is in freedrive mode.

        Returns:
            bool: True if the robot is in freedrive mode, False otherwise.
        """
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Set the freedrive mode of the robot.

        Args:
            enable (bool): True to enable freedrive mode, False to disable it.
        """
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        if enable and not self._free_drive:
            self._free_drive = True
            self.r_inter.StartDrag()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.r_inter.StopDrag()

    def get_observations(self) -> Dict[str, np.ndarray]:
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        joints = self.get_joint_state()
        pos_rot= self.get_eef_pose()
        gripper_pos = np.array([joints[-1]])
        print("ee_pos_quat for obs saving:", pos_rot)
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_rot,  # TODO: this is pos_rot actually
            "gripper_position": gripper_pos,
        }


    def get_obs(self) -> Dict[str, np.ndarray]:
        # 获取关节状态
        joints = self.get_joint_state()
        # 初始化位置和四元数
        pos_quat = self.get_eef_pose()
        # 获取夹爪位置
        gripper_pos = np.array([joints[-1]])
        # 返回观测值
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }

    def get_DI_state(self, index):
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        return int(self.r_inter.DI(index).split('{')[1].split('}')[0])

    def set_do_status(self, which_do):
        assert not self.robot_is_err, f"{self.robot_ip}: error!"
        self.r_inter.DO(which_do[0], which_do[1])
        return 1


def main():
    dobot = DobotRobot("192.168.5.2", no_gripper=False)
    dobot.set_do_status([1, 0])
    dobot.set_do_status([2, 0])
    dobot.set_do_status([3, 0])

    joint_state = np.array([76, -15, 79, -6, -100, -88, 1.0])
    robot_joints_angle = joint_state[:6]  # ��λ������
    robot_joints = [np.deg2rad(robot_joint) for robot_joint in robot_joints_angle]

    print("joint_radians:", robot_joints)

    # fk via forward kinematics function
    coef = 1 # right hand
    claw = claw_width(joint_state[-1])
    claw *= coef
    pos, rot = forward_kinematics(*robot_joints, claw)
    pos = pos * 1000  # Convert mm to m
    rot = R.from_matrix(rot).as_euler('xyz', degrees=True)
    eef_state = np.concatenate((pos, rot, [joint_state[-1]]))

    # fk via dobot api
    # eef_state = dobot.get_fk(joint_state)
    # eef_state = np.concatenate((eef_state, [joint_state[-1]]))
    print("eef_state for testing:", eef_state)

    # dobot.command_joint_state(joint_state)eef_state
    # pos_rot = dobot.get_eef_pose()
    # print("get_eef_pose:", pos_rot)

    # eef_state = np.array([-27, -383, 415, 170, -32, -105, 1.0])
    dobot.command_eef_state(eef_state)



    # obs = dobot.get_observations()
    # print("get_observations:", obs)

    # while 1:
    #     dobot.get_joint_state()
    # dobot = DobotRobot("192.168.5.2", no_gripper=False)
    # set_light(dobot, "red", 0)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
