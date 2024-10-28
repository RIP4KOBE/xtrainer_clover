import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
import os

class ImageSaver:
    def __init__(self):
        self.rgb_callback_flag = False
        self.depth_callback_flag = False

        # Image save rate parameter
        self.save_rate = 1  # Default to 1 Hz
        self.rate = self.save_rate

        # Last save times
        self.last_rgb_save_time = 0
        self.last_depth_save_time = 0

        # Subscribers
        self.rgb_img_sub = threading.Thread(target=self.rgb_callback)
        self.depth_img_sub = threading.Thread(target=self.depth_callback)

        # Directories for saving images
        self.rgb_save_dir = "/home/zhuoli/xtrainer_clover/curigpt_ros/assets/img"
        self.depth_save_dir = "/home/zhuoli/xtrainer_clover/curigpt_ros/assets/img"

        # Create directories if they don't exist
        if not os.path.exists(self.rgb_save_dir):
            os.makedirs(self.rgb_save_dir)
        if not os.path.exists(self.depth_save_dir):
            os.makedirs(self.depth_save_dir)

        self.image_count = 0

        # Timer for checking the reception of images
        self.timer = threading.Timer(1, self.check_images_received)

    def rgb_callback(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device("419122270852")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)

        # Start streaming
        pipeline.start(config)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                self.rgb_callback_flag = True
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())

                current_time = time.time()
                if (current_time - self.last_rgb_save_time) > 1 / self.save_rate:
                    # Save images
                    cv2.imwrite(os.path.join(self.rgb_save_dir, "local_rgb_test_coaster.png"), color_image)
                    self.image_count += 1
                    self.last_rgb_save_time = time.time()
                # Show images
                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Stop streaming
            pipeline.stop()

    def depth_callback(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        pipeline.start(config)

        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Save images
                cv2.imwrite(os.path.join(self.depth_save_dir, "realtime_depth_test.png"), depth_colormap)
                self.image_count += 1
                self.last_depth_save_time = time.time()

                # Show images
                cv2.imshow('RealSense', depth_colormap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Stop streaming
            pipeline.stop()

    def check_images_received(self):
        if not self.rgb_callback_flag and self.depth_callback_flag:
            print("Waiting for RGB-D image")

        # Reset flags after checking
        self.rgb_callback_flag = False
        self.depth_callback_flag = False

def save_images_realsense():
    image_saver = ImageSaver()
    image_saver.rgb_img_sub.start()
    # image_saver.depth_img_sub.start()
    image_saver.timer.start()

if __name__ == '__main__':
    save_images_realsense()
