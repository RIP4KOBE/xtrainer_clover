import numpy as np
import cv2
import pickle
from ModelTrain.dp.utils import get_eef_delta


# image
# image_imcode = np.load("/home/dobot/projects/datasets/dataset1_cleanDish/collect_data/20240511111035/leftImg/1.npy")
# image = cv2.imdecode(np.asarray(image_imcode, dtype="uint8"), cv2.IMREAD_COLOR)
# cv2.imshow("1", image)
# cv2.waitKey(1000)

# data
with open("/media/zhuoli/8ECE-77DB/xtrainer/Datasets/DP/dp_plate_wiping_eef_delta_20250617/collect_data/20250617212159/2025-06-17T21-22-03-775598.pkl", "rb") as f:
    data_single = pickle.load(f)
    delta_action = get_eef_delta(
        data_single["ee_pos_quat"][:6], data_single["control"][:6]
    )
    print("eef delta left arm:", delta_action)
