import numpy as np
import cv2
import os
import pickle
from ModelTrain.dp.utils import get_eef_delta
from ModelTrain.dp.dataset import normalize_6d_pose, unnormalize_6d_pose, unnormalize_data, normalize_data
from ModelTrain.dp.utils import rotation_vector_to_sixd



# image
# image_imcode = np.load("/home/dobot/projects/datasets/dataset1_cleanDish/collect_data/20240511111035/leftImg/1.npy")
# image = cv2.imdecode(np.asarray(image_imcode, dtype="uint8"), cv2.IMREAD_COLOR)
# cv2.imshow("1", image)
# cv2.waitKey(1000)

# stats
with open(
        "/home/zhuoli/dobot_xtrainer/model/dp_plate_wiping_eef_absolute_6d_normalization_20250619/0619_161449_A4Jk-camera=012-identity=False-repr=IP-oh=1-ah=8-ph=16-prefix=None-do=0.0-imgos=32-wd=1e-05-use_ddim=True-binarize_touch=False-eef6d/stats.pkl",
        "rb") as f:
    stats = pickle.load(f)


# data sequence
data = []
dir_path = "/home/zhuoli/dobot_xtrainer/datasets/dp_plate_wiping_eef_delta_20250617/collect_data/20250617210028"

for filename in os.listdir(dir_path):
    filepath = os.path.join(dir_path, filename)
    if filename.endswith('.pkl') or filename.endswith('.pickle'):  # 检查pickle文件
        with open(filepath, 'rb') as f:
            data_single = pickle.load(f)
            eef_action_rotvec = data_single["control"]
            lef_act_pos = eef_action_rotvec[:3]
            lef_act_rot = eef_action_rotvec[3:6]
            left_gripper_act = eef_action_rotvec[6]
            right_act_pos = eef_action_rotvec[7:10]
            right_act_rot = eef_action_rotvec[10:13]
            right_gripper_act = eef_action_rotvec[13]

            # Convert rotation vector to 6D representation
            lef_act_rot_6d = rotation_vector_to_sixd(lef_act_rot)
            right_act_rot_6d = rotation_vector_to_sixd(right_act_rot)

            eef_action_6d = np.concatenate(
                (lef_act_pos, lef_act_rot_6d, [left_gripper_act], right_act_pos, right_act_rot_6d, [right_gripper_act])
            )
            data.append(eef_action_6d)


# Normalize the 6D pose
data_array = np.stack(data, axis=0)
eef_action_normalized = normalize_data(data_array, stats["action"])
eef_action_unormalized = unnormalize_data(eef_action_normalized, stats["action"])
eef_action_6d_normalized = normalize_6d_pose(data_array, stats["action"])
eef_action_6d_unormalized = unnormalize_6d_pose(eef_action_6d_normalized, stats["action"])

print("original eef action:", data_array)
print("normalized eef action:", eef_action_6d_normalized)
print("unormalized eef action:", eef_action_6d_unormalized)




    # print("eef delta left arm:", delta_action)
