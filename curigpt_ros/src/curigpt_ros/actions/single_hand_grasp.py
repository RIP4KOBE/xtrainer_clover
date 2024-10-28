#!/usr/bin/env python
# import rospy
# from std_msgs.msg import String
# from geometry_msgs.msg import Pose, TransformStamped
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
# from curigpt_ros.srv import ExecuteGroupPose, ExecuteGroupPoseRequest
from curigpt_ros.utils import ros_utils
# import ExecuteGroupPose, ExecuteGroupPoseRequest
from curigpt_ros.utils.transform import Rotation, Transform
from curigpt_ros.utils.ros_utils import *


import json

def transform_coordinates(camera_manipulation_point):

    tf_tree_cam = ros_utils.TransformTree()

    camera_manipulation_point = camera_manipulation_point.flatten()

    T_camcolor_grasp = Transform(Rotation.from_quat([0, 0, 0, 1]),
                                         [camera_manipulation_point[0], camera_manipulation_point[1],
                                          camera_manipulation_point[2]])

    # T_camlink_camcolor = tf_tree_cam.lookup(
    #     "camera_link", "camera_color_optical_frame", rospy.Time(0), rospy.Duration(0.1)
    # )
    # print("T_camlink_camcolor: ", to_transform_msg(T_camlink_camcolor))

    T_camlink_camcolor = Transform(Rotation.from_quat([-0.500, 0.500, -0.500, 0.500]),
                                         [0, 0, 0])

    # transformation of hand-eye calibration
    T_base_camlink = Transform(Rotation.from_quat([0.400276, -0.04094002, 0.1665276, 0.900206]),
                               [-0.014567831, -0.3578937, 0.19013431])


    T_base_grasp = T_base_camlink * T_camlink_camcolor * T_camcolor_grasp

    return T_base_grasp.translation


def publish_grasp_pose_to_service(grasp_pose, log_info):

    # Wait for the service to be available
    rospy.wait_for_service('/panda_left/execute_create_ptp_cartesian_trajectory')
    rospy.loginfo("Trajectory execution service is available")

    try:
        # Create a service client
        service_client = rospy.ServiceProxy('/panda_left/execute_create_ptp_cartesian_trajectory', ExecuteGroupPose)

        # Create the service request
        service_request = ExecuteGroupPoseRequest()

        # Fill in the request
        service_request.header.seq = 0
        service_request.header.stamp = rospy.Time.now()  # Or rospy.Time(0) for current time
        service_request.header.frame_id = ''  # Set if needed
        service_request.group_name = ''  # Set if needed
        service_request.goal_type = 0  # Set if needed

        # Set the position and orientation from the grasp_pose
        service_request.goal.position = grasp_pose.position
        service_request.goal.orientation = grasp_pose.orientation

        service_request.tolerance = 8
        service_request.constraint = ''  # Set if needed

        # Call the service
        service_response = service_client(service_request)

        # Handle the response
        if not service_response.SUCCEEDED:
            rospy.loginfo(log_info)
        else:
            rospy.loginfo("Service call failed.")

    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)


def softhand_grasp(synergy_control_parameters):
    # define softhand publisher
    softhand_grasp_pub = rospy.Publisher('/qbhand2m_chain/control/qbhand2m2_synergies_trajectory_controller/command', JointTrajectory, queue_size=10)

    traj_l = JointTrajectory()
    traj_l.header.stamp = rospy.Time.now()
    traj_l.header.frame_id = " "
    traj_l.joint_names = ["qbhand2m2_manipulation_joint", "qbhand2m2_synergy_joint"]
    point = JointTrajectoryPoint()
    point.positions = [synergy_control_parameters[1], synergy_control_parameters[0]]
    point.time_from_start = rospy.Duration(1)
    traj_l.points.append(point)

    for i in range(50):
        softhand_grasp_pub.publish(traj_l)
        rospy.loginfo("softhand grasp execution...")
        # 短暂休眠，以确保消息被发送并且不会过载接收方
        rospy.sleep(0.1)  # 休眠100毫秒

    # rospy.sleep(10)


def grasp_and_place(camera_grasp_point, camera_place_point):

    # rospy.init_node('grasp_and_place_node')
    rospy.loginfo("Starting grasp_and_place operation")
    rate = rospy.Rate(1)

    # Transform the coordinates from the bounding box to robot base coordinates
    robot_grasp_point = transform_coordinates(camera_grasp_point)
    robot_place_point = transform_coordinates(camera_place_point)

    # define pre-grasp pose
    initial_pose = Pose()
    initial_pose.position.x = 0.12925132
    initial_pose.position.y = -0.059268
    initial_pose.position.z = 1.03635
    initial_pose.orientation.x = 0.714494
    initial_pose.orientation.y = -0.25879
    initial_pose.orientation.z = 0.4299753
    initial_pose.orientation.w = -0.487487

    # define grasp pose
    grasp_pose = Pose()
    grasp_pose.position.x = robot_grasp_point[0]
    grasp_pose.position.y = robot_grasp_point[1] + 0.05
    grasp_pose.position.z = robot_grasp_point[2] + 0.12
    grasp_pose.orientation.x = -0.58267
    grasp_pose.orientation.y = 0.39935
    grasp_pose.orientation.z = -0.30119
    grasp_pose.orientation.w = 0.64053

    # define grasp-place middle pose
    middle_pose = Pose()
    middle_pose.position.x = robot_grasp_point[0] - 0.2
    middle_pose.position.y = robot_grasp_point[1] - 0.2
    middle_pose.position.z = robot_grasp_point[2] + 0.15
    middle_pose.orientation.x = -0.58267
    middle_pose.orientation.y = 0.39935
    middle_pose.orientation.z = -0.30119
    middle_pose.orientation.w = 0.64053

    # define place pose
    place_pose = Pose()
    place_pose.position.x = robot_place_point[0] - 0.2
    place_pose.position.y = robot_place_point[1] - 0.2
    place_pose.position.z = robot_place_point[2] + 0.13
    place_pose.orientation.x = -0.58267
    place_pose.orientation.y = 0.39935
    place_pose.orientation.z = -0.30119
    place_pose.orientation.w = 0.64053

    # define softhand grasp synergy parameters
    close_synergy_control_parameters = [1.0, 0.0]
    open_synergy_control_parameters = [0.0, 0.0]

    # move to pre-grasp pose
    # publish_grasp_pose_to_service(initial_pose, "Moving to pre-grasp pose...")

    # move to grasp pose
    publish_grasp_pose_to_service(grasp_pose, "Moving to grasp pose...")
    # rospy.sleep(3)

    # close softhand
    # softhand_grasp(close_synergy_control_parameters)

    # move to middle pose
    publish_grasp_pose_to_service(middle_pose, "Moving to middle pose...")

    # move to place pose
    publish_grasp_pose_to_service(place_pose, "Moving to place pose...")
    # rospy.sleep(3)

    # open softhand
    # softhand_grasp(open_synergy_control_parameters)

    # move back
    publish_grasp_pose_to_service(initial_pose, "Moving back...")

    rospy.loginfo("Grasp and place operation completed")

def grasp_and_give(camera_grasp_point):

    # rospy.init_node('grasp_and_give_node')
    rospy.loginfo("Starting grasp_and_give operation")
    rate = rospy.Rate(1)

    # Transform the coordinates from the bounding box to robot base coordinates
    robot_grasp_point = transform_coordinates(camera_grasp_point)

    print("robot_grasp_point: ", robot_grasp_point)

    # define initial pose
    initial_pose = Pose()
    initial_pose.position.x = 0.12925132
    initial_pose.position.y = -0.059268
    initial_pose.position.z = 1.03635
    initial_pose.orientation.x = 0.714494
    initial_pose.orientation.y = -0.25879
    initial_pose.orientation.z = 0.4299753
    initial_pose.orientation.w = -0.487487

    # define pre-grasp pose
    # pre_grasp_pose = Pose()
    # pre_grasp_pose.position.x = robot_grasp_point[0] - 0.05
    # pre_grasp_pose.position.y = robot_grasp_point[1] - 0.05
    # pre_grasp_pose.position.z = robot_grasp_point[2] + 0.18
    # pre_grasp_pose.orientation.x = -0.58267
    # pre_grasp_pose.orientation.y = 0.39935
    # pre_grasp_pose.orientation.z = -0.30119
    # pre_grasp_pose.orientation.w = 0.64053

    # define grasp pose
    grasp_pose = Pose()
    grasp_pose.position.x = robot_grasp_point[0]
    grasp_pose.position.y = robot_grasp_point[1]
    grasp_pose.position.z = robot_grasp_point[2] + 0.165
    grasp_pose.orientation.x = -0.58267
    grasp_pose.orientation.y = 0.39935
    grasp_pose.orientation.z = -0.30119
    grasp_pose.orientation.w = 0.64053

    # define grasp pose
    # grasp_pose = Pose()
    # grasp_pose.position.x = robot_grasp_point[0]
    # grasp_pose.position.y = robot_grasp_point[1]
    # grasp_pose.position.z = robot_grasp_point[2] + 0.18
    # grasp_pose.orientation.x = -0.303952
    # grasp_pose.orientation.y = 0.6563
    # grasp_pose.orientation.z = 0.00729
    # grasp_pose.orientation.w = 0.6905127

    # # define grasp-give middle pose
    # middle_pose = Pose()
    # middle_pose.position.x = robot_grasp_point[0] - 0.2
    # middle_pose.position.y = robot_grasp_point[1] - 0.2
    # middle_pose.position.z = robot_grasp_point[2] + 0.18
    # middle_pose.orientation.x = -0.303952
    # middle_pose.orientation.y = 0.6563
    # middle_pose.orientation.z = 0.00729
    # middle_pose.orientation.w = 0.6905127

    # define grasp-give middle pose
    middle_pose = Pose()
    middle_pose.position.x = robot_grasp_point[0] - 0.2
    middle_pose.position.y = robot_grasp_point[1] - 0.2
    middle_pose.position.z = robot_grasp_point[2] + 0.18
    middle_pose.orientation.x = -0.58267
    middle_pose.orientation.y = 0.39935
    middle_pose.orientation.z = -0.30119
    middle_pose.orientation.w = 0.64053

    # define give pose
    give_pose = Pose()
    give_pose.position.x = 0.5114361
    give_pose.position.y = -0.36646
    give_pose.position.z = 0.75690
    give_pose.orientation.x = 0.742553
    give_pose.orientation.y =  0.0214382
    give_pose.orientation.z = 0.574130089
    give_pose.orientation.w = -0.34428

    # define softhand grasp synergy parameters
    close_synergy_control_parameters = [1.0, 0.0]
    open_synergy_control_parameters = [0.0, 0.0]

    # move to initial pose
    # publish_grasp_pose_to_service(initial_pose, "Moving to initial pose...")

    # move to pre-grasp pose
    # publish_grasp_pose_to_service(pre_grasp_pose, "Moving to pre-grasp pose...")


    # move to grasp pose
    publish_grasp_pose_to_service(grasp_pose, "Moving to grasp pose...")

    # move to middle pose
    publish_grasp_pose_to_service(middle_pose, "Moving to middle pose...")
    # rospy.sleep(3)

    # close softhand
    # softhand_grasp(close_synergy_control_parameters)

    # move to give pose
    publish_grasp_pose_to_service(give_pose, "Moving to give pose...")
    # rospy.sleep(3)

    # move back
    publish_grasp_pose_to_service(initial_pose, "Moving back to pre-grasp pose...")

    # open softhand
    # softhand_grasp(open_synergy_control_parameters)


    rospy.loginfo("Grasp and give operation completed")


def grasp_and_place_demo():

    # rospy.init_node('grasp_and_place_node')
    rospy.loginfo("Starting grasp_and_place operation")
    rate = rospy.Rate(1)

    # define pre-grasp pose
    initial_pose = Pose()
    initial_pose.position.x = 0.12925132
    initial_pose.position.y = -0.059268
    initial_pose.position.z = 1.03635
    initial_pose.orientation.x = 0.714494
    initial_pose.orientation.y = -0.25879
    initial_pose.orientation.z = 0.4299753
    initial_pose.orientation.w = -0.487487

    # define grasp pose
    grasp_pose = Pose()
    grasp_pose.position.x = 0.4631561
    grasp_pose.position.y = -0.04459
    grasp_pose.position.z = 0.61642358
    grasp_pose.orientation.x = -0.539049
    grasp_pose.orientation.y = 0.5799498
    grasp_pose.orientation.z = -0.127830
    grasp_pose.orientation.w = 0.597279

    # define grasp-place middle pose
    middle_pose = Pose()
    middle_pose.position.x = 0.365970
    middle_pose.position.y = -0.226880
    middle_pose.position.z = 0.598135026
    middle_pose.orientation.x = -0.4987873
    middle_pose.orientation.y = 0.6009702
    middle_pose.orientation.z = -0.20929944
    middle_pose.orientation.w = 0.5884213


    # define place pose
    place_pose = Pose()
    place_pose.position.x = 0.51700
    place_pose.position.y = -0.27812
    place_pose.position.z = 0.45883624
    place_pose.orientation.x = -0.54149480
    place_pose.orientation.y = 0.56266139
    place_pose.orientation.z = -0.20938017
    place_pose.orientation.w = 0.588519

    # define softhand grasp synergy parameters
    close_synergy_control_parameters = [1.0, 0.0]
    open_synergy_control_parameters = [0.0, 0.0]

    # move to pre-grasp pose
    publish_grasp_pose_to_service(initial_pose, "Moving to pre-grasp pose...")

    # move to grasp pose
    publish_grasp_pose_to_service(grasp_pose, "Moving to grasp pose...")
    # rospy.sleep(3)

    # close softhand
    # softhand_grasp(close_synergy_control_parameters)

    # move to middle pose
    publish_grasp_pose_to_service(middle_pose, "Moving to middle pose...")

    # move to place pose
    publish_grasp_pose_to_service(place_pose, "Moving to place pose...")
    # rospy.sleep(3)

    # open softhand
    # softhand_grasp(open_synergy_control_parameters)

    # move back
    publish_grasp_pose_to_service(initial_pose, "Moving back...")

    rospy.loginfo("Grasp and place operation completed")


def grasp_and_give_demo():
    # rospy.init_node('grasp_and_give_node')
    rospy.loginfo("Starting grasp_and_give operation")
    rate = rospy.Rate(1)

    # define pre-grasp pose
    initial_pose = Pose()
    initial_pose.position.x = 0.12925132
    initial_pose.position.y = -0.059268
    initial_pose.position.z = 1.03635
    initial_pose.orientation.x = 0.714494
    initial_pose.orientation.y = -0.25879
    initial_pose.orientation.z = 0.4299753
    initial_pose.orientation.w = -0.487487

    # define grasp pose1
    grasp_pose = Pose()
    grasp_pose.position.x = 0.4405777
    grasp_pose.position.y = 0.03181093
    grasp_pose.position.z = 0.6261
    grasp_pose.orientation.x = 0.9210253
    grasp_pose.orientation.y = -0.369937
    grasp_pose.orientation.z = 0.12184
    grasp_pose.orientation.w = 0.0034452




# define grasp pose2
    # grasp_pose2 = Pose()
    # grasp_pose2.position.x = 0.50793483
    # grasp_pose2.position.y = 0.1019290
    # grasp_pose2.position.z = 0.5642419
    # grasp_pose2.orientation.x = 0.8749206
    # grasp_pose2.orientation.y = -0.2777139
    # grasp_pose2.orientation.z = 0.32665304
    # grasp_pose2.orientation.w = 0.2251369


    # define give pose
    give_pose = Pose()
    give_pose.position.x = 0.5114361
    give_pose.position.y = -0.36646
    give_pose.position.z = 0.75690
    give_pose.orientation.x = 0.742553
    give_pose.orientation.y =  0.0214382
    give_pose.orientation.z = 0.574130089
    give_pose.orientation.w = -0.34428


    # define softhand grasp synergy parameters
    close_synergy_control_parameters = [1.0, 0.0]
    open_synergy_control_parameters = [0.0, 0.0]

    # move to pre-grasp pose
    publish_grasp_pose_to_service(initial_pose, "Moving to pre-grasp pose...")

    # move to grasp pose
    publish_grasp_pose_to_service(grasp_pose, "Moving to grasp pose...")
    # rospy.sleep(3)

    # close softhand
    # softhand_grasp(close_synergy_control_parameters)


    # move to pre-grasp pose
    # publish_grasp_pose_to_service(initial_pose, "Moving to pre-grasp pose...")
    # rospy.sleep(25)

    # move to grasp pose2
    # publish_grasp_pose_to_service(grasp_pose2, "Moving to grasp pose...")
    # rospy.sleep(3)

    # move to give pose
    # publish_grasp_pose_to_service(give_pose, "Moving to give pose...")
    # rospy.sleep(3)

    # move back
    publish_grasp_pose_to_service(initial_pose, "Moving back to pre-grasp pose...")

    # open softhand
    # softhand_grasp(open_synergy_control_parameters)

    rospy.loginfo("Grasp and give operation completed")


def grasp_handover_give(arg1, arg2):

    rospy.init_node('grasp_handover_place_node')
    rospy.loginfo("Starting grasp_handover_place operation")
    rate = rospy.Rate(1)

    # Transform the coordinates from the bounding box to robot base coordinates
    grasp_point = transform_coordinates(arg1)
    place_point = transform_coordinates(arg2)

    # define pre-grasp pose
    initial_pose = Pose()
    initial_pose.position.x = 0.452
    initial_pose.position.y = 0.020
    initial_pose.position.z = 0.957
    initial_pose.orientation.x = 0.861
    initial_pose.orientation.y = -0.239
    initial_pose.orientation.z = 0.385
    initial_pose.orientation.w = -0.232

    # define grasp pose
    grasp_pose = Pose()
    grasp_pose.position.x = grasp_point[0]
    grasp_pose.position.y = grasp_point[1]
    grasp_pose.position.z = grasp_point[2]
    grasp_pose.orientation.x = 0.861
    grasp_pose.orientation.y = -0.239
    grasp_pose.orientation.z = 0.385
    grasp_pose.orientation.w = -0.232

    # define place pose
    place_pose = Pose()
    place_pose.position.x = place_point[0]
    place_pose.position.y = place_point[1]
    place_pose.position.z = place_point[2]
    place_pose.orientation.x = 0.861
    place_pose.orientation.y = -0.239
    place_pose.orientation.z = 0.385
    place_pose.orientation.w = -0.232

    # define softhand grasp synergy parameters
    close_synergy_control_parameters = [1.0, 0.0]
    open_synergy_control_parameters = [0.0, 0.0]

    # move to pre-grasp pose
    publish_grasp_pose_to_service(initial_pose, "Moving to pre-grasp pose...")

    # move to grasp pose
    publish_grasp_pose_to_service(grasp_pose, "Moving to grasp pose...")
    # rospy.sleep(3)

    # close softhand
    softhand_grasp(close_synergy_control_parameters)

    # move to place pose
    publish_grasp_pose_to_service(place_pose, "Moving to post-grasp pose...")
    # rospy.sleep(3)

    # open softhand
    softhand_grasp(open_synergy_control_parameters)

    # move back
    publish_grasp_pose_to_service(initial_pose, "Moving to pre-grasp pose...")

    rospy.loginfo("Grasp and place operation completed")


if __name__ == '__main__':
    rospy.init_node('grasp_and_place_node')
    grasp_and_give_demo()
    # grasp_and_place_demo()
    # transform_coordinates([0, 0, 0])
    rospy.spin()
    rospy.loginfo("Shutting down grasp_and_place operation")