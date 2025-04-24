import torch
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from ModelTrain.dp.vis_utils import filter_points_by_bounds
from ModelTrain.dp.utils import get_config
from sklearn.cluster import MeanShift
from scipy.spatial.transform import Rotation as R
from scripts.manipulate_utils import load_ini_data_camera
from huggingface_hub import hf_hub_download
from segment_anything import SamAutomaticMaskGenerator, build_sam_vit_b
from dobot_control.cameras.realsense_camera import RealSenseCamera, get_device_ids
from vis_utils import pixel_to_3d_points
import numpy as np
import cv2
import json


class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)

        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)

        # for each mask, cluster in feature space to get meaningful regions, and uske their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat,
                                                                                                  masks)
        print("candidate_keypoints length before filter:", len(candidate_keypoints))
        # 打印 x, y, z 坐标上的最小值
        print("candidate_keypoints (x min):", candidate_keypoints[:, 0].min())
        print("candidate_keypoints (y min):", candidate_keypoints[:, 1].min())
        print("candidate_keypoints (z min):", candidate_keypoints[:, 2].min())

        # 打印 x, y, z 坐标上的最大值
        print("candidate_keypoints (x max):", candidate_keypoints[:, 0].max())
        print("candidate_keypoints (y max):", candidate_keypoints[:, 1].max())
        print("candidate_keypoints (z max):", candidate_keypoints[:, 2].max())

        # exclude keypoints that are outside of the workspace
        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        print("candidate_keypoints length after filter:", len(candidate_keypoints))
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]


        # merge close points by clustering in cartesian space
        merged_indices = self._merge_clusters(candidate_keypoints)
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]


        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]


        # project keypoints to image space
        projected = self._project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids, masks,
                                                   features_flat)
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        masks = [m['segmentation'] for m in masks]

        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size

        transformed_rgb = cv2.resize(rgb, (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]

        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }

        return transformed_rgb, rgb, points, masks, shape_info

    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        projected = rgb.copy()
        # overlay keypoints on the image
        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # draw a box
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                          (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                          (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # draw text
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(
            self.device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w,
                                                    -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),
                                                # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(
            0)  # float32 [H, W, feature_dim]
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[
            -1])  # float32 [H*W, feature_dim]
        return features_flat

    def _cluster_features(self, points, features_flat, masks):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        for rigid_group_id, binary_mask in enumerate(masks):
            # ignore mask that is too large
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                        feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # cluster features to get meaningful regions
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices


if __name__ == "__main__":
    # camera init
    device_ids = get_device_ids()
    print(f"Found {len(device_ids)} devices: ", device_ids)

    camera_dict = load_ini_data_camera()
    rs_list = [RealSenseCamera(flip=True, device_id=camera_dict["top"]),
               RealSenseCamera(flip=False, device_id=camera_dict["left"]),
               RealSenseCamera(flip=True, device_id=camera_dict["right"])]

    # Read images from cameras
    base_rgb, base_depth = rs_list[0].read()
    print("base_depth type from realsense:", type(base_depth))
    base_rgb = base_rgb[:, :, ::-1]
    cv2.imshow("0", base_rgb)
    cv2.imshow("1", base_depth)
    cv2.waitKey(1000)  # 显示 1 秒后继续
    cv2.destroyAllWindows()

    np.savetxt("base_depth_values.txt", base_depth, fmt="%.3f")


    # Get camera intrinsic and extrinsic parameters
    depth_intr, _ = rs_list[0].get_parameters()
    print("depth_intr:", depth_intr)

    # T_color_to_base
    quat_color_to_base = [0.8883237745164563, 0.0013136423315819848, -0.4583985850907289, 0.02738399458585889]
    t_color_to_base = np.array([-0.4944436068757155, -0.5615340320230741, 1.01093569778022])
    rot_color_to_base = R.from_quat(quat_color_to_base).as_matrix()
    T_color_to_base = np.vstack((
        np.hstack((rot_color_to_base, t_color_to_base.reshape(3, 1))),
        [0, 0, 0, 1]
    ))

    # # T_depth_to_color
    # quat_depth_to_color = [-0.5, 0.5, -0.5, -0.499]
    # t_depth_to_color = np.array([0.0, 0.0, 0.0])  # no translation
    # rot_depth_to_color = R.from_quat(quat_depth_to_color).as_matrix()
    # T_depth_to_color = np.vstack((
    #     np.hstack((rot_depth_to_color, t_depth_to_color.reshape(3, 1))),
    #     [0, 0, 0, 1]
    # ))

    # == 3. 合并变换：T_depth_to_base =
    # depth_extr = T_color_to_base @ T_depth_to_color
    depth_extr = T_color_to_base


    print("depth_extr:\n", depth_extr)

    # get points and masks
    # points
    points = pixel_to_3d_points(base_depth, depth_intr, depth_extr)

    # 打印 x, y, z 坐标上的最小值
    print("points (x min):", points[:, 0].min())
    print("points (y min):", points[:, 1].min())
    print("points (z min):", points[:, 2].min())

    # 打印 x, y, z 坐标上的最大值
    print("points (x max):", points[:, 0].max())
    print("points (y max):", points[:, 1].max())
    print("points (z max):", points[:, 2].max())
    # Start SAM
    print("start mask generation")
    sam_chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
    sam_model = build_sam_vit_b(checkpoint=sam_chkpt_path)
    sam_model.to("cuda")
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    # Generate masks
    masks = mask_generator.generate(base_rgb)

    # Ensure base_rgb is in correct format for OpenCV
    if not isinstance(base_rgb, np.ndarray):
        base_rgb = base_rgb.cpu().numpy()
    if base_rgb.dtype != np.uint8:
        base_rgb = (base_rgb * 255).astype(np.uint8) if base_rgb.max() <= 1.0 else base_rgb.astype(np.uint8)
    base_rgb = np.ascontiguousarray(base_rgb)

    # Draw masks
    for mask in masks:
        mask_area = np.uint8(mask['segmentation']) * 255
        contours, _ = cv2.findContours(mask_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(base_rgb, [cnt], -1, (0, 255, 0), 3)

    # Display result
    cv2.imshow("mask", base_rgb)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # predictor = SamPredictor(build_sam(checkpoint="checkpoints/sam_vit_b_01ec64.pth"))
    # predictor.set_image(base_rgb)
    # masks, _, _ = predictor.predict( < input_prompts >)

    # initialize keypoint proposer
    keypoint_config = get_config(config_path="/home/zhuoli/xtrainer_clover/configs/keypoint_config.yaml")
    keypoint_proposer = KeypointProposer(keypoint_config['keypoint_proposer'])

    candidate_keypoints, projected_img = keypoint_proposer.get_keypoints(base_rgb, points, masks)
    print("Candidate Keypoints:", candidate_keypoints)
    print("Projected Image Shape:", projected_img.shape)

    # Visualize the projected image
    visualize= True
    if visualize:
        cv2.imshow('Projected Image', projected_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    # Save metadata as JSON
    metadata = {
        'init_keypoint_positions': candidate_keypoints.tolist(),  # Ensure numpy arrays are converted
        'num_keypoints': len(candidate_keypoints)
    }

    with open('metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
