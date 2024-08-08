from scripts.manipulate_utils import load_ini_data_camera
from dobot_control.cameras.realsense_camera import RealSenseCamera, get_device_ids
import numpy as np
import cv2


# camera init
device_ids = get_device_ids()
print(f"Found {len(device_ids)} devices: ", device_ids)

camera_dict = load_ini_data_camera()
rs_list = [RealSenseCamera(flip=True, device_id=camera_dict["top"]),
           RealSenseCamera(flip=False, device_id=camera_dict["left"]),
           RealSenseCamera(flip=True, device_id=camera_dict["right"])]
show_canvas = np.zeros((480, 640*3, 3), dtype=np.uint8)

while 1:
    for i in range(len(rs_list)):
        _img, _ = rs_list[i].read()
        _img = _img[:, :, ::-1]
        show_canvas[:, int(640*i):int(640*(i+1))] = np.asarray(_img, dtype="uint8")
    cv2.imshow("0", show_canvas)
    cv2.waitKey(1)
