# import rospy
# from curigpt_ros.actions.single_hand_grasp import grasp_and_place, grasp_and_give, grasp_handover_give
# from curigpt_ros.utils.vis_utils import plot_image_with_bbox, get_spatial_coordinates, vis_spatial_point
# from curigpt_ros.srv import ExecuteGroupPose, ExecuteGroupPoseRequest
import open3d as o3d

def process_robot_actions(action_response, rgb_img, depth_img):
    # Map each action to its corresponding function
    action_map = {
        "grasp_and_place": grasp_and_place,
        "grasp_and_give": grasp_and_give,
        "grasp_handover_give": grasp_handover_give
    }

    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=360, fx=345.9, fy=346.0, cx=322.7, cy=181.3) # Camera intrinsics: fx, fy, cx, cy
    cam_manip_points = []

    if action_response[0]["action"] in action_map:
        action_func = action_map[action_response[0]["action"]]
        print("Processing action:", action_response[0]["action"])

        # Visualize the bounding box to check if they are correct
        plot_image_with_bbox(rgb_img, action_response)

        if action_response[0]["action"] in ["grasp_and_give", "grasp_handover_give"]:
            grasp_bbox = action_response[0]["parameters"]["arg1"]["bbox_coordinates"]
            spatial_grasp_point = get_spatial_coordinates(grasp_bbox, rgb_img, depth_img, cam_intrinsics)
            cam_manip_points.append(spatial_grasp_point)
            # vis_spatial_point(cam_manip_points, rgb_img, depth_img, cam_intrinsics)
            action_func(spatial_grasp_point)

        elif action_response[0]["action"] == 'grasp_and_place':
            grasp_bbox = action_response[0]["parameters"]["arg1"]["bbox_coordinates"]
            place_bbox = action_response[0]["parameters"]["arg2"]["bbox_coordinates"]
            spatial_grasp_point = get_spatial_coordinates(grasp_bbox, rgb_img, depth_img, cam_intrinsics)
            spatial_place_point = get_spatial_coordinates(place_bbox, rgb_img, depth_img, cam_intrinsics)
            cam_manip_points.append(spatial_grasp_point)
            cam_manip_points.append(spatial_place_point)
            # Visualize the spatial points to check if they are correct
            # vis_spatial_point(cam_manip_points, rgb_img, depth_img, cam_intrinsics)
            action_func(spatial_grasp_point, spatial_place_point)



    else:
        print("Unknown action:", action_response['action'])


def publish_waypoint_to_service(waypoint, log_info):

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
        service_request.goal.position = waypoint.position
        service_request.goal.orientation = waypoint.orientation

        service_request.tolerance = 5
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