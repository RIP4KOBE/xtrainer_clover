#!/usr/bin/env python
# import rospy
# from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image, ImageDraw
import cv2
import pyrealsense2 as rs
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import open3d as o3d

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.rgb_callback_flag = False
        self.depth_callback_flag = False

        # Image save rate parameter
        self.save_rate = rospy.get_param('~save_rate', 1)  # Default to 1 Hz
        self.rate = rospy.Rate(self.save_rate)

        # Last save times
        self.last_rgb_save_time = rospy.Time.now()
        self.last_depth_save_time = rospy.Time.now()

        # Subscribers
        self.rgb_img_sub = rospy.Subscriber("/camera/color/image_raw", RosImage, self.rgb_callback)
        self.depth_img_sub = rospy.Subscriber("/camera/depth/image_raw", RosImage, self.depth_callback)

        # Directories for saving images
        self.rgb_save_dir = "assets/img/realtime_rgb"
        self.depth_save_dir = "assets/img/realtime_depth"

        # Create directories if they don't exist
        if not os.path.exists(self.rgb_save_dir):
            os.makedirs(self.rgb_save_dir)
        if not os.path.exists(self.depth_save_dir):
            os.makedirs(self.depth_save_dir)

        self.image_count = 0

        # Timer for checking the reception of images
        self.timer = rospy.Timer(rospy.Duration(1), self.check_images_received)

    def rgb_callback(self, data):
        self.rgb_callback_flag = True  # Set flag to true when image is received

        current_time = rospy.Time.now()
        if (current_time - self.last_rgb_save_time).to_sec() > 1 / self.save_rate:
            try:
                # Convert ROS image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                img_filename = os.path.join(self.rgb_save_dir, "realtime_rgb.png")
                cv2.imwrite(img_filename, cv_image)
                # print(f"Saved RGB image to {img_filename}")
                self.image_count += 1
                self.last_rgb_save_time = current_time
            except CvBridgeError as e:
                print(e)

    def depth_callback(self, data):
        self.depth_callback_flag = True  # Set flag to true when image is received

        current_time = rospy.Time.now()
        if (current_time - self.last_depth_save_time).to_sec() > 1.0 / self.save_rate:
            try:
                img_filename = os.path.join(self.depth_save_dir, "realtime_depth.png")

                # Convert ROS depth image message to OpenCV image
                cv_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
                cv2.imwrite(img_filename, cv_image)

                # Open and save the file in binary write mode
                # with open(img_filename, 'wb') as file:
                #     file.write(data.data)

                # print(f"Saved Depth image to {img_filename}")
                self.image_count += 1
                self.last_depth_save_time = current_time
            except CvBridgeError as e:
                print(e)

    def check_images_received(self, event):
        if not self.rgb_callback_flag and self.depth_callback_flag:
            print("Waiting for RGB-D image")

        # Reset flags after checking
        self.rgb_callback_flag = False
        self.depth_callback_flag = False


def save_color_image_rs():
    """
    Capture and save color images from the RealSense camera.

    Parameters:
        folder_path (str): folder to save the images.
        save_interval (int): interval in seconds to save images.
    """

    folder_path = "../../assets/img/realtime_rgb"

    # ensure the folder exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # set up the RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # start the pipeline
    pipeline.start(config)
    save_interval = 1
    save_times = 2

    try:
        for i in range(save_times):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("No color frame captured")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            img_name = "realtime_rgb.png"
            cv2.imwrite(os.path.join(folder_path, img_name), color_image)
            time.sleep(save_interval)

        print(f"RGB image saved")

    finally:
        # stop the pipeline and close the OpenCV windows
        pipeline.stop()
        cv2.destroyAllWindows()


def save_images_gemini():
    """
    Capture color and depth images from the Gemini camera.

    Parameters:
        folder_path (str): folder to save the images.
        save_interval (int): interval in seconds to save images.
    """
    image_saver = ImageSaver()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


def add_bbox_patch(ax, draw, bbox, color, w, h, caption):
    """
    Add a bounding box patch to the plot.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to add the patch to.
        draw (ImageDraw.Draw): The ImageDraw object to draw on the image.
        bbox (list): The bounding box coordinates.
        color (str): The color of the bounding box.
        w (int): The width of the image.
        h (int): The height of the image.
        caption (str): The caption to annotate the bounding box with.
    """
    # Draw the bounding box on the image
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # Annotate the bounding box with a caption
    ax.text(x1, y1, caption, color=color, weight='bold', fontsize=8,
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.1'))


def plot_image_with_bbox(rgb_img_path, action_response):
    """
    Parse the bbox coordinates from the response of CuriGPT and plot them on the image.

    Parameters:
    image_path (str): The path to the image file.
    action_response (str): The action_response from the multimodal large language model.
    """

    # load the image.
    image = Image.open(rgb_img_path)
    draw = ImageDraw.Draw(image)
    # get the width and height od the image size
    w, h = image.size

    # create a plot.
    fig, ax = plt.subplots()

    # display the image.
    ax.imshow(image)

    # go through each action and plot the bounding boxes.
    if action_response is not None:
        for action in action_response:
            bbox1 = action['parameters']['arg1']['bbox_coordinates']
            caption1 = action['parameters']['arg1']['description']

            # check if arg2 exists
            if 'arg2' in action['parameters']:
                bbox2 = action['parameters']['arg2']['bbox_coordinates']
                caption2 = action['parameters']['arg2']['description']

                add_bbox_patch(ax, draw, bbox1, 'red', w, h, caption1)
                add_bbox_patch(ax, draw, bbox2, 'blue', w, h, caption2)

            else:
                add_bbox_patch(ax, draw, bbox1, 'red', w, h, caption1)

        # show the plot with the bounding boxes.
        plt.show()


def get_spatial_coordinates(bbox, rgb_img, depth_img, cam_intrinsics):
    """
    Calculate 3D coordinates of the center of a bounding box using the depth map and camera intrinsics.

    :param bbox: Tuple (x, y, width, height) defining the bounding box.
    :param depth_map: 2D array where each element is the depth in meters from the camera.
    :param camera_intrinsics: Camera intrinsic matrix typically shaped (3, 3).
    :return: Numpy array containing the 3D coordinates (X, Y, Z) in meters.
    """

    # Open the depth image
    color_raw = o3d.io.read_image(rgb_img)
    depth_raw = o3d.io.read_image(depth_img)

    rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    color_array = np.asarray(rgbd_img.color)
    depth_array = np.asarray(rgbd_img.depth)

    # assert the size of color image and depth image are both 848*480, otherwise through an error
    assert color_array.shape[0] == depth_array.shape[0] and color_array.shape[1] == depth_array.shape[1], "The size of color image and depth image are not the same."

    img_width, img_hight = depth_array.shape[1], depth_array.shape[0]


    # Assuming the depth is scaled as depth in millimeters
    # depth_array = depth_array.astype(np.float32) / 1000.0

    # define the camera intrinsics
    camera_intrinsics_matrix = cam_intrinsics.intrinsic_matrix

    # Extract the bounding box center
    x1, y1, x2, y2 = (int(bbox[0] / 1000 * img_width), int(bbox[1] / 1000 * img_hight), int(bbox[2] / 1000 *
                                                                                            img_width),
                      int(bbox[3] / 1000 * img_hight))

    print(x1, y1, x2, y2)

    # calculate the center of the bounding box
    x_center = int((x1 + x2) / 2)
    y_center = int((y1 + y2) / 2)

    # Get the depth value at the center of the bounding box
    z = depth_array[y_center, x_center]

    print("Depth value at center of bounding box:", z)

    # Create a vector for the pixel coordinates including the depth
    pixel_vector = np.array([x_center * z, y_center * z, z])
    print("Pixel Vector:", pixel_vector)
    pixel_vector = pixel_vector.reshape(3, 1)  # Ensuring the pixel_vector is a column vector

    # Compute the inverse of the camera intrinsic matrix
    try:
        # Attempt to invert the camera intrinsics matrix
        camera_intrinsics_inv = np.linalg.inv(camera_intrinsics_matrix)
        # print("Inverted Camera Intrinsics Matrix:\n", camera_intrinsics_inv)
    except np.linalg.LinAlgError as e:
        print("Error inverting camera intrinsics:", e)
        return None

    # Calculate the 3D coordinates
    spatial_coordinates = camera_intrinsics_inv.dot(pixel_vector)
    print("World Coordinates (X, Y, Z):", spatial_coordinates)

    return spatial_coordinates


def create_point_cloud_from_rgbd(rgb_image_path, depth_image_path, camera_intrinsics):
    # Load RGB and depth images
    color_raw = o3d.io.read_image(rgb_image_path)
    depth_raw = o3d.io.read_image(depth_image_path)

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsics
    )

    # Translate the point cloud for a better visualization (optional)
    # pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    return pcd

def vis_spatial_point(spatial_coordinates_list, rgb_img, depth_img, cam_intrinsics):

    if not spatial_coordinates_list:
        print("No points or colors provided for visualization.")
        return

    # create the scene point cloud
    scene_pcd = create_point_cloud_from_rgbd(rgb_img, depth_img, cam_intrinsics)

    # prepare visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(scene_pcd)
    colors_list = [[1, 0, 0], [0, 1, 0]]  # RGB colors: red and green

    # Iterate over the list of coordinates and colors
    for point, color in zip(spatial_coordinates_list, colors_list):
        # Create a sphere to represent the highlighted point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # Adjust the radius as needed
        sphere.paint_uniform_color(color)  # Set color
        translation_vector = np.reshape(point, (3, 1))
        sphere.translate(translation_vector)

        # Add geometry to visualizer
        vis.add_geometry(sphere)

    # Run the visualizer
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    bbox =[343, 761, 516, 888]
    rgb_img = "assets/img/realtime_rgb/realtime_rgb.png"
    depth_img = "assets/img/realtime_depth/realtime_depth.png"
    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=640, height=360, fx=345.9, fy=346.0, cx=322.7, cy=181.3)

    get_spatial_coordinates(bbox, rgb_img, depth_img, cam_intrinsics)

    spatial_grasp_point = [[-0.08743192,  0.20586329,  0.62099999]]
    vis_spatial_point(spatial_grasp_point, rgb_img, depth_img, cam_intrinsics)
