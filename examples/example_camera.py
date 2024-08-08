import cv2
from dobot_control.cameras.realsense_camera import RealSenseCamera, get_device_ids

device_ids = get_device_ids()
print(f"Found {len(device_ids)} devices: ", device_ids)
rs = RealSenseCamera(flip=True, device_id=device_ids[0])
while 1:
    image, _ = rs.read()
    image = image[:, :, ::-1]
    cv2.imshow("demo", image)
    cv2.waitKey(1)
