import sys
import os
# 获取根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from dataclasses import dataclass
import tyro
from dobot_control.robots.dobot import DobotRobot
from dobot_control.robots.robot import BimanualRobot, PrintRobot
from dobot_control.robots.robot_node import ZMQServerRobot


@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"


def launch_robot_server(args: Args):
    port = args.robot_port
    _robot_l = DobotRobot(robot_ip="192.168.5.1", robot_number=2)  # IP of the left hand robotic arm
    _robot_r = DobotRobot(robot_ip="192.168.5.2", robot_number=2)  # IP of the rigth hand robotic arm
    robot = BimanualRobot(_robot_l, _robot_r)
    server = ZMQServerRobot(robot, port=port, host=args.hostname)
    print(f"Starting robot server on port {port}")
    server.serve()


def main(args):
    launch_robot_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
