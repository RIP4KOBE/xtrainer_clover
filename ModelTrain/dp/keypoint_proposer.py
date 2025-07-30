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
from vis_utils import pixel_to_world_points, visualize_and_pick_point, compute_world_coordinates_from_depth, pixel_to_camera_points
import numpy as np
import cv2
import json
import open3d as o3d

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
        # load handeye calibration
        handeye_trans = np.array(config['handeye_translation'])
        handeye_quat = np.array(config['handeye_quaternion'])
        self.T_link_to_base = np.vstack((
            np.hstack((R.from_quat(handeye_quat).as_matrix(), handeye_trans.reshape(3, 1))),
            [0, 0, 0, 1]
        ))

    def get_keypoints(self, rgb, points, masks, rotate_text_180=False):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)

        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)

        # for each mask, cluster in feature space to get meaningful regions, and uske their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat,
                                                                                                  masks)

        # exclude keypoints that are outside of the workspace
        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
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
                                                   features_flat, rotate_text_180)
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

    # def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
    #     projected = rgb.copy()
    #     # overlay keypoints on the image
    #     for keypoint_count, pixel in enumerate(candidate_pixels):
    #         displayed_text = f"{keypoint_count}"
    #         text_length = len(displayed_text)
    #         # draw a box
    #         box_width = 30 + 10 * (text_length - 1)
    #         box_height = 30
    #         cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
    #                       (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
    #         cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
    #                       (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
    #         # draw text
    #         org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
    #         color = (255, 0, 0)
    #         cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    #         keypoint_count += 1
    #     return projected

    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat,
                                  rotate_text_180=False):
        projected = rgb.copy()
        height, width = projected.shape[:2]

        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            box_width = 18 + 6 * (text_length - 1)
            box_height = 18

            # ---- Step 1: Draw white box with black border directly on projected image ----
            top_left = (pixel[1] - box_width // 2, pixel[0] - box_height // 2)
            bottom_right = (pixel[1] + box_width // 2, pixel[0] + box_height // 2)

            # Draw filled white rectangle
            cv2.rectangle(projected, top_left, bottom_right, (255, 255, 255), -1)
            # Draw black border
            cv2.rectangle(projected, top_left, bottom_right, (0, 0, 0), 2)

            # ---- Step 2: Create a patch for the text only ----
            patch = np.ones((box_height, box_width, 3), dtype=np.uint8) * 255  # white background
            text_size = cv2.getTextSize(displayed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (box_width - text_size[0]) // 2
            text_y = (box_height + text_size[1]) // 2

            # Draw text onto patch
            cv2.putText(patch, displayed_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # ---- Step 3: Rotate text patch if needed ----
            if rotate_text_180:
                patch = cv2.rotate(patch, cv2.ROTATE_180)

            # ---- Step 4: Overlay patch (text only) onto white box ----
            y1 = max(0, top_left[1])
            y2 = min(width, bottom_right[1])
            x1 = max(0, top_left[0])
            x2 = min(height, bottom_right[0])

            # Ensure the patch fits entirely within image bounds
            if 0 <= top_left[1] < width - box_width and 0 <= top_left[0] < height - box_height:
                projected[top_left[1]:top_left[1] + box_height, top_left[0]:top_left[0] + box_width] = patch

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

            if np.sum(binary_mask) < 10:
                print(f"Skipping mask {rigid_group_id}: too few pixels")
                continue

            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            if torch.isnan(obj_features_flat).any():
                print(f"Skipping mask {rigid_group_id}: NaN in obj_features_flat")
                continue

            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])

            X = features_pca

            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch = (feature_points_torch - feature_points_torch.min(0)[0]) / (
                        feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            if torch.isnan(feature_points_torch).any():
                print(f"Skipping mask {rigid_group_id}: NaN in feature_points_torch")
                continue
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

    def run(self, visualize_points=False, visualize_projection=False, check_value=False):
        # camera init
        camera_dict = load_ini_data_camera()
        rs_list = [RealSenseCamera(flip=False, device_id=camera_dict["top"])]

        # get depth image, camera intrinsic and extrinsic parameters
        print("start 3D keypoints extraction")
        align_depth = True
        if align_depth:
            base_rgb, base_depth, color_intr, depth_intr = rs_list[0].read_alignment()

            # Transformation from camera_color_optical frame to camera_link frame
            quat_color_to_link = [-0.499, 0.499, -0.498, 0.503]
            t_color_to_link = np.array([-0.000, 0.015, -0.000])  # no translation
            rot_color_to_link = R.from_quat(quat_color_to_link).as_matrix()
            T_color_to_link = np.vstack((
                np.hstack((rot_color_to_link, t_color_to_link.reshape(3, 1))),
                [0, 0, 0, 1]
            ))
            # T_color_to_link = np.linalg.inv(T_color_to_link)
            extrinsics = self.T_link_to_base @ T_color_to_link

        else:
            base_rgb, base_depth = rs_list[0].read()
            depth_intr, _ = rs_list[
                0].get_parameters()  # intrinsic of depth camera (camera_depth_frame or camera_depth_optical_frame)
            extrinsics = self.T_link_to_base  # camera_link is aligned with camera_depth_frame in realsense d435i

        # cv2.imshow("0", base_rgb)
        # # cv2.imshow("1", base_depth)
        # cv2.waitKey(3000)  # 显示 1 秒后继续
        # cv2.destroyAllWindows()
        # cv2.imwrite('/home/zhuoli/xtrainer_clover/configs/base_rgb.png', base_rgb)

        # get points
        points = pixel_to_world_points(base_depth, depth_intr, extrinsics)

        if check_value:
            check_points = points.reshape(-1, 3)
            print("points shape for checking:", check_points.shape)
            print("points (x min):", check_points[:, 0].min())
            print("points (y min):", check_points[:, 1].min())
            print("points (z min):", check_points[:, 2].min())
            print("points (x max):", check_points[:, 0].max())
            print("points (y max):", check_points[:, 1].max())
            print("points (z max):", check_points[:, 2].max())

        if visualize_points:
            picked_points = visualize_and_pick_point(points, base_rgb)

        # get masks
        sam_chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_b_01ec64.pth")
        sam_model = build_sam_vit_b(checkpoint=sam_chkpt_path)
        sam_model.to("cuda")
        mask_generator = SamAutomaticMaskGenerator(sam_model)
        masks = mask_generator.generate(base_rgb)

        candidate_keypoints, projected_img = self.get_keypoints(base_rgb, points, masks, rotate_text_180=True)
        print("Candidate Keypoints:", candidate_keypoints)

        if visualize_projection:
            cv2.imshow('Projected Image', projected_img)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            projected_img = cv2.rotate(projected_img, cv2.ROTATE_180)
            cv2.imwrite('/home/zhuoli/xtrainer_clover/configs/projected_image.png', projected_img)

        # save keypoints as metadata
        candidate_keypoints = candidate_keypoints.tolist()
        metadata = {
            'keypoint_positions': candidate_keypoints,  # Ensure numpy arrays are converted
            'num_keypoints': len(candidate_keypoints)
        }

        with open('/home/zhuoli/xtrainer_clover/configs/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        return candidate_keypoints

if __name__ == "__main__":
    keypoint_config = get_config(config_path="/home/zhuoli/xtrainer_clover/configs/keypoint_config.yaml")
    keypoint_proposer = KeypointProposer(keypoint_config['keypoint_proposer'])
    keypoints = keypoint_proposer.run(visualize_projection=True)
    print("Keypoints:", keypoints[0])

