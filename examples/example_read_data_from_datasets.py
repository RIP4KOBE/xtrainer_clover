import numpy as np
import cv2
import pickle


# image
image_imcode = np.load("/home/dobot/projects/datasets/dataset1_cleanDish/collect_data/20240511111035/leftImg/1.npy")
image = cv2.imdecode(np.asarray(image_imcode, dtype="uint8"), cv2.IMREAD_COLOR)
cv2.imshow("1", image)
cv2.waitKey(1000)

# data
with open("/home/dobot/projects/datasets/dataset1_cleanDish/collect_data/20240511111035/observation/1.pkl", "rb") as f:
    data_single = pickle.load(f)
    print(data_single['joint_positions'])
    print(data_single['joint_velocities'])
    print(data_single['control'])