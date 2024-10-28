#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose, TransformStamped
from pydub import AudioSegment
from pydub.playback import play
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from curigpt_ros.srv import ExecuteGroupPose, ExecuteGroupPoseRequest
from curigpt_ros.utils import ros_utils
# import ExecuteGroupPose, ExecuteGroupPoseRequest
from curigpt_ros.utils.transform import Rotation, Transform
from curigpt_ros.utils.ros_utils import *
from curgpt_ros.utils.action_utils import publish_waypoint_to_service
from curigpt_ros.models.audio_assistant import play_audio
import json
import threading


def curi_welcome():

    rospy.loginfo("Starting welcome operation")
    rate = rospy.Rate(5)

    # define audio paths
    welcome_audio = "assets/chat_audio/welcome_audio/welcome_audio.mp3"
    hand_shake_audio = "assets/chat_audio/welcome_audio/hand_shake_audio.mp3"
    interactive_demo_audio = "assets/chat_audio/welcome_audio/interactive_demo_audio.mp3"

    # define multiple audio threads
    welcome_audio_thread = threading.Thread(target=play_audio, args=(welcome_audio,))
    hand_shake_audio_thread = threading.Thread(target=play_audio, args=(hand_shake_audio,))
    interactive_demo_audio_thread = threading.Thread(target=play_audio, args=(interactive_demo_audio,))

    # define wave hand waypoints
    wave_pose1 = Pose()
    wave_pose1.position.x = 0.12925132
    wave_pose1.position.y = -0.059268
    wave_pose1.position.z = 1.03635
    wave_pose1.orientation.x = 0.714494
    wave_pose1.orientation.y = -0.25879
    wave_pose1.orientation.z = 0.4299753
    wave_pose1.orientation.w = -0.487487

    wave_pose2 = Pose()
    wave_pose2.position.x = 0.12925132
    wave_pose2.position.y = -0.059268
    wave_pose2.position.z = 1.03635
    wave_pose2.orientation.x = 0.714494
    wave_pose2.orientation.y = -0.25879
    wave_pose2.orientation.z = 0.4299753
    wave_pose2.orientation.w = -0.487487

    wave_pose3 = Pose()
    wave_pose3.position.x = 0.12925132
    wave_pose3.position.y = -0.059268
    wave_pose3.position.z = 1.03635
    wave_pose3.orientation.x = 0.714494
    wave_pose3.orientation.y = -0.25879
    wave_pose3.orientation.z = 0.4299753
    wave_pose3.orientation.w = -0.487487

    wave_pose4 = Pose()
    wave_pose4.position.x = 0.12925132
    wave_pose4.position.y = -0.059268
    wave_pose4.position.z = 1.03635
    wave_pose4.orientation.x = 0.714494
    wave_pose4.orientation.y = -0.25879
    wave_pose4.orientation.z = 0.4299753
    wave_pose4.orientation.w = -0.487487

    # store wave waypoints into a list
    wave_poses = [wave_pose1, wave_pose2, wave_pose3, wave_pose4]

    # start welcome audio and publish wave hand waypoints
    welcome_audio_thread.start()
    for wave_pose in wave_poses:
        publish_waypoint_to_service(wave_pose, "Waving hand...")
        # rate.sleep()

    # define hand shake waypoints
    hand_shake_pose1 = Pose()
    hand_shake_pose1.position.x = 0.12925132
    hand_shake_pose1.position.y = -0.059268
    hand_shake_pose1.position.z = 1.03635
    hand_shake_pose1.orientation.x = 0.714494
    hand_shake_pose1.orientation.y = -0.25879
    hand_shake_pose1.orientation.z = 0.4299753
    hand_shake_pose1.orientation.w = -0.487487

    hand_shake_pose2 = Pose()
    hand_shake_pose2.position.x = 0.12925132
    hand_shake_pose2.position.y = -0.059268
    hand_shake_pose2.position.z = 1.03635
    hand_shake_pose2.orientation.x = 0.714494
    hand_shake_pose2.orientation.y = -0.25879
    hand_shake_pose2.orientation.z = 0.4299753
    hand_shake_pose2.orientation.w = -0.487487

    hand_shake_pose3 = Pose()
    hand_shake_pose3.position.x = 0.12925132
    hand_shake_pose3.position.y = -0.059268
    hand_shake_pose3.position.z = 1.03635
    hand_shake_pose3.orientation.x = 0.714494
    hand_shake_pose3.orientation.y = -0.25879
    hand_shake_pose3.orientation.z = 0.4299753
    hand_shake_pose3.orientation.w = -0.487487

    # store hand shake waypoints into a list
    hand_shake_poses = [hand_shake_pose1, hand_shake_pose2, hand_shake_pose3]

    # start hand shake audio and publish hand shake hand waypoints
    hand_shake_audio_thread.start()
    for hand_shake_pose in hand_shake_poses:
        publish_waypoint_to_service(hand_shake_pose, "Shaking hand...")

    interactive_demo_audio_thread.start()
    rate.sleep()

    rospy.signal_shutdown("Shutting down ROS node.")
    welcome_audio_thread.join()
    hand_shake_audio_thread.join()
    interactive_demo_audio_thread.join()
    rospy.loginfo("CURI welcome operation completed")

if __name__ == '__main__':
    rospy.init_node('curi_welcome_node')
    curi_welcome()