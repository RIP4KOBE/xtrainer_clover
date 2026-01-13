import os.path

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
from matplotlib import pyplot as plt

class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.save_dir = self.config['save_dir']
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

    def visualize_dino_features(self, rgb, features_flat, shape_info):
        """
        Visualize DINOv2 features using PCA
        Args:
            rgb: original RGB image [H, W, 3]
            features_flat: flattened features [H*W, feature_dim]
            shape_info: dict containing img_h, img_w
        Returns:
            feature_rgb: RGB visualization of features [H, W, 3]
        """
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']

        # Step 1: PCA降维到3维
        features_flat = features_flat.double()
        u, s, v = torch.pca_lowrank(features_flat, q=3, center=True)
        features_pca = torch.mm(features_flat, v[:, :3])  # [H*W, 3]

        # Step 2: 归一化到 [0, 1]
        features_pca = (features_pca - features_pca.min(0)[0]) / (
                features_pca.max(0)[0] - features_pca.min(0)[0] + 1e-8
        )

        # Step 3: 转换为图像格式
        feature_rgb = features_pca.reshape(img_h, img_w, 3).cpu().numpy()
        feature_rgb = (feature_rgb * 255).astype(np.uint8)

        # Step 4: 可选：与原图混合
        alpha = 0.6  # 特征图权重
        blended = cv2.addWeighted(rgb, 1 - alpha, feature_rgb, alpha, 0)

        return feature_rgb, blended

    def visualize_mask_features(self, rgb, features_flat, masks):
        """
        为每个mask区域可视化DINOv2特征
        Args:
            rgb: 原始RGB图像 [H, W, 3]
            features_flat: 扁平化特征 [H*W, feature_dim]
            masks: SAM分割的mask列表
        Returns:
            feature_map: 特征可视化图 [H, W, 3]
        """
        feature_map = np.zeros_like(rgb)
        h, w = rgb.shape[:2]

        # 为每个mask分配随机颜色（可选，用于区分不同mask）
        np.random.seed(42)

        valid_mask_count = 0
        for mask_id, binary_mask in enumerate(masks):
            # 获取该mask的特征
            mask_flat = binary_mask.reshape(-1)
            mask_features = features_flat[mask_flat]

            # 跳过太小的mask
            if len(mask_features) < 10:
                print(f"Skipping mask {mask_id}: too few pixels ({len(mask_features)})")
                continue

            try:
                # PCA降维到3维
                mask_features = mask_features.double()
                u, s, v = torch.pca_lowrank(mask_features, q=3, center=True)
                features_pca = torch.mm(mask_features, v[:, :3])

                # 归一化到 [0, 1]
                features_pca = (features_pca - features_pca.min(0)[0]) / (
                        features_pca.max(0)[0] - features_pca.min(0)[0] + 1e-8
                )

                # 转换为RGB值 [0, 255]
                mask_rgb = (features_pca.cpu().numpy() * 255).astype(np.uint8)

                # 映射回图像
                feature_map[binary_mask] = mask_rgb
                valid_mask_count += 1

            except Exception as e:
                print(f"Error processing mask {mask_id}: {e}")
                continue

        print(f"Successfully visualized {valid_mask_count}/{len(masks)} masks")
        return feature_map

    def _display_mask_features(self, rgb, mask_feature_map, masks):
        """
        显示和保存mask特征可视化结果
        """
        import matplotlib.pyplot as plt

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 原始图像
        axes[0, 0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')

        # Mask边界叠加
        mask_overlay = rgb.copy()
        for mask in masks:
            color = np.random.randint(0, 255, 3).tolist()
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(mask_overlay, contours, -1, color, 2)
        axes[0, 1].imshow(cv2.cvtColor(mask_overlay, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'SAM Masks ({len(masks)} masks)', fontsize=14)
        axes[0, 1].axis('off')

        # 特征可视化
        axes[1, 0].imshow(mask_feature_map)
        axes[1, 0].set_title('DINOv2 Features per Mask', fontsize=14)
        axes[1, 0].axis('off')

        # 混合图像
        alpha = 0.5
        blended = cv2.addWeighted(rgb, 1 - alpha, mask_feature_map, alpha, 0)
        axes[1, 1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Blended (Original + Features)', fontsize=14)
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 保存
        save_path = os.path.join(self.save_dir, 'mask_features_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

        plt.show()
        plt.close()

        # 单独保存特征图
        cv2.imwrite(
            os.path.join(self.save_dir, 'mask_features_only.png'),
            cv2.cvtColor(mask_feature_map, cv2.COLOR_RGB2BGR)
        )

    def visualize_features_tsne(self, rgb, features_flat, perplexity=30,
                                n_iter=1000, learning_rate=200):
        """
        使用t-SNE将高维特征降维到2D并可视化为RGB图像
        Args:
            rgb: 原始RGB图像 [H, W, 3]
            features_flat: 扁平化特征 [H*W, feature_dim]
            perplexity: t-SNE困惑度参数 (5-50)
            n_iter: 迭代次数
            learning_rate: 学习率
        Returns:
            tsne_map: t-SNE可视化图 [H, W, 3]
        """
        from sklearn.manifold import TSNE
        import time

        h, w = rgb.shape[:2]
        n_samples = h * w

        print(f"Running t-SNE on {n_samples} samples with {features_flat.shape[1]} dimensions...")
        print(f"Parameters: perplexity={perplexity}, n_iter={n_iter}, learning_rate={learning_rate}")

        # 转换为numpy
        features_np = features_flat.cpu().numpy()

        # 可选: 降采样以加速 (如果图像太大)
        if n_samples > 50000:
            print(f"Warning: Large image ({n_samples} pixels). Consider downsampling.")
            downsample_rate = int(np.sqrt(n_samples / 50000))
            indices = np.arange(0, n_samples, downsample_rate)
            features_sampled = features_np[indices]
            print(f"Downsampled to {len(indices)} samples (rate: 1/{downsample_rate})")
        else:
            features_sampled = features_np
            indices = None

        # 运行t-SNE
        start_time = time.time()
        tsne = TSNE(
            n_components=3,  # 降维到3D用于RGB可视化
            perplexity=perplexity,
            n_iter=n_iter,
            learning_rate=learning_rate,
            random_state=42,
            verbose=1
        )
        features_tsne = tsne.fit_transform(features_sampled)
        elapsed = time.time() - start_time
        print(f"t-SNE completed in {elapsed:.2f} seconds")

        # 如果进行了降采样，需要插值回原始分辨率
        if indices is not None:
            from scipy.interpolate import griddata

            # 创建坐标网格
            yi, xi = np.unravel_index(indices, (h, w))
            points = np.column_stack((yi, xi))

            # 目标网格
            grid_y, grid_x = np.mgrid[0:h, 0:w]

            # 对每个通道进行插值
            features_tsne_full = np.zeros((h * w, 3))
            for i in range(3):
                features_tsne_full[:, i] = griddata(
                    points, features_tsne[:, i],
                    (grid_y.ravel(), grid_x.ravel()),
                    method='linear'
                )
            features_tsne = features_tsne_full

        # 归一化到 [0, 1]
        features_tsne = (features_tsne - features_tsne.min(axis=0)) / (
                features_tsne.max(axis=0) - features_tsne.min(axis=0) + 1e-8
        )

        # 转换为RGB值 [0, 255] 并reshape
        tsne_map = (features_tsne * 255).astype(np.uint8).reshape(h, w, 3)

        return tsne_map

    def _display_tsne(self, rgb, tsne_map, save_name='tsne_visualization.png'):
        """
        显示和保存t-SNE可视化结果
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')

        # t-SNE特征图
        axes[1].imshow(tsne_map)
        axes[1].set_title('t-SNE Feature Visualization', fontsize=14)
        axes[1].axis('off')

        # 混合图像
        alpha = 0.5
        blended = cv2.addWeighted(rgb, 1 - alpha, tsne_map, alpha, 0)
        axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Blended (alpha={alpha})', fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()

        # 保存
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE visualization to {save_path}")

        plt.show()
        plt.close()

        # 单独保存特征图
        cv2.imwrite(
            os.path.join(self.save_dir, 'tsne_features_only.png'),
            cv2.cvtColor(tsne_map, cv2.COLOR_RGB2BGR)
        )

    def get_keypoints(self, rgb, points, masks, rotate_text_180=False, visualize_dino_features=False, vis_mask_features=False, visualize_tsne=False):
        # preprocessing
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)

        # get features
        features_flat = self._get_features(transformed_rgb, shape_info)

        if visualize_dino_features:
            # 可视化特征
            feature_rgb, blended = self.visualize_dino_features(
                rgb, features_flat, shape_info
            )

            # 显示结果
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(feature_rgb)
            axes[1].set_title('DINOv2 Features (PCA)')
            axes[1].axis('off')

            axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            axes[2].set_title('Blended')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'feature_visualization.png'),
                        dpi=150, bbox_inches='tight')
            plt.show()

            # 保存特征图
            cv2.imwrite(
                os.path.join(self.save_dir, 'dino_features.png'),
                cv2.cvtColor(feature_rgb, cv2.COLOR_RGB2BGR)
            )

        if vis_mask_features:
            # 可视化每个mask的特征
            mask_feature_map = self.visualize_mask_features(
                rgb, features_flat, masks
            )

            # 显示结果
            self._display_mask_features(rgb, mask_feature_map, masks)

        if visualize_tsne:
            # 基础t-SNE可视化
            tsne_map = self.visualize_features_tsne(
                rgb, features_flat,
                perplexity=30,
                n_iter=1000
            )
            self._display_tsne(rgb, tsne_map)

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

    # def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat,
    #                               rotate_text_180=False):
    #     projected = rgb.copy()
    #     height, width = projected.shape[:2]
    #
    #     for keypoint_count, pixel in enumerate(candidate_pixels):
    #         displayed_text = f"{keypoint_count}"
    #         text_length = len(displayed_text)
    #         box_width = 18 + 6 * (text_length - 1)
    #         box_height = 18
    #
    #         # ---- Step 1: Draw white box with black border directly on projected image ----
    #         top_left = (pixel[1] - box_width // 2, pixel[0] - box_height // 2)
    #         bottom_right = (pixel[1] + box_width // 2, pixel[0] + box_height // 2)
    #
    #         # Draw filled white rectangle
    #         cv2.rectangle(projected, top_left, bottom_right, (255, 255, 255), -1)
    #         # Draw black border
    #         cv2.rectangle(projected, top_left, bottom_right, (0, 0, 0), 2)
    #
    #         # ---- Step 2: Create a patch for the text only ----
    #         patch = np.ones((box_height, box_width, 3), dtype=np.uint8) * 255  # white background
    #         text_size = cv2.getTextSize(displayed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    #         text_x = (box_width - text_size[0]) // 2
    #         text_y = (box_height + text_size[1]) // 2
    #
    #         # Draw text onto patch
    #         cv2.putText(patch, displayed_text, (text_x, text_y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #
    #         # ---- Step 3: Rotate text patch if needed ----
    #         if rotate_text_180:
    #             patch = cv2.rotate(patch, cv2.ROTATE_180)
    #
    #         # ---- Step 4: Overlay patch (text only) onto white box ----
    #         y1 = max(0, top_left[1])
    #         y2 = min(width, bottom_right[1])
    #         x1 = max(0, top_left[0])
    #         x2 = min(height, bottom_right[0])
    #
    #         # Ensure the patch fits entirely within image bounds
    #         if 0 <= top_left[1] < width - box_width and 0 <= top_left[0] < height - box_height:
    #             projected[top_left[1]:top_left[1] + box_height, top_left[0]:top_left[0] + box_width] = patch
    #
    #     return projected

    # def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat,
    #                               rotate_text_180=False):
    #     projected = rgb.copy()
    #     height, width = projected.shape[:2]
    #
    #     for keypoint_count, pixel in enumerate(candidate_pixels):
    #         displayed_text = f"{keypoint_count}"
    #         text_length = len(displayed_text)
    #         box_width = 18 + 6 * (text_length - 1)
    #         box_height = 18
    #
    #         # 确保像素坐标是整数
    #         y, x = int(pixel[0]), int(pixel[1])
    #
    #         # 计算框的边界，确保在图像范围内
    #         x1 = max(0, x - box_width // 2)
    #         y1 = max(0, y - box_height // 2)
    #         x2 = min(width, x + box_width // 2)
    #         y2 = min(height, y + box_height // 2)
    #
    #         # 计算实际框的宽高（可能因边界裁剪而改变）
    #         actual_width = x2 - x1
    #         actual_height = y2 - y1
    #
    #         # 只有当框有足够的尺寸时才继续
    #         if actual_width <= 0 or actual_height <= 0:
    #             continue
    #
    #         # 绘制白色框
    #         cv2.rectangle(projected, (x1, y1), (x2, y2), (255, 255, 255), -1)
    #         # 绘制黑色边框
    #         cv2.rectangle(projected, (x1, y1), (x2, y2), (0, 0, 0), 2)
    #
    #         # 创建文本补丁，与实际框大小相匹配
    #         patch = np.ones((actual_height, actual_width, 3), dtype=np.uint8) * 255
    #
    #         # 计算文本尺寸
    #         text_size = cv2.getTextSize(displayed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    #
    #         # 居中放置文本
    #         text_x = (actual_width - text_size[0]) // 2
    #         text_y = (actual_height + text_size[1]) // 2
    #
    #         # 确保文本位置不为负
    #         text_x = max(0, text_x)
    #         text_y = max(text_size[1], text_y)
    #
    #         # 在补丁上绘制文本
    #         cv2.putText(patch, displayed_text, (text_x, text_y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    #
    #         # 根据需要旋转文本
    #         if rotate_text_180:
    #             patch = cv2.rotate(patch, cv2.ROTATE_180)
    #
    #         # 将补丁覆盖到图像上
    #         try:
    #             projected[y1:y2, x1:x2] = patch
    #         except ValueError as e:
    #             # 打印错误，帮助调试尺寸不匹配问题
    #             print(f"Error at keypoint {keypoint_count}: {e}")
    #             print(f"Patch shape: {patch.shape}, Target area: ({y2 - y1}, {x2 - x1})")
    #             # 尝试调整patch大小以匹配目标区域
    #             if y2 - y1 > 0 and x2 - x1 > 0:
    #                 resized_patch = cv2.resize(patch, (x2 - x1, y2 - y1))
    #                 projected[y1:y2, x1:x2] = resized_patch
    #
    #     return projected

    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat,
                                  rotate_text_180=False):
        projected = rgb.copy()
        height, width = projected.shape[:2]

        for keypoint_count, pixel in enumerate(candidate_pixels):
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)

            # Improvement: Dynamically adjust box size based on text length
            # Provide more space for double digits and above
            if text_length == 1:
                box_width = 18
            else:
                # For double digits and above, add 8 pixels width per additional digit
                box_width = 24 + 8 * (text_length - 2)

            box_height = 18  # Keep height unchanged or adjust as needed

            # Ensure pixel coordinates are integers
            y, x = int(pixel[0]), int(pixel[1])

            # Calculate box boundaries, ensuring they're within image bounds
            x1 = max(0, x - box_width // 2)
            y1 = max(0, y - box_height // 2)
            x2 = min(width, x + box_width // 2)
            y2 = min(height, y + box_height // 2)

            # Calculate actual box width and height (may change due to boundary clipping)
            actual_width = x2 - x1
            actual_height = y2 - y1

            # Only proceed if the box has sufficient size
            if actual_width <= 0 or actual_height <= 0:
                continue

            # Draw white box
            cv2.rectangle(projected, (x1, y1), (x2, y2), (255, 255, 255), -1)
            # Draw black border
            cv2.rectangle(projected, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Create text patch that matches the actual box size
            patch = np.ones((actual_height, actual_width, 3), dtype=np.uint8) * 255

            # Improvement: Adjust font size based on text length
            font_scale = 0.7 if text_length == 1 else 0.6

            # Calculate text size
            text_size = cv2.getTextSize(displayed_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]

            # Center the text
            text_x = (actual_width - text_size[0]) // 2
            text_y = (actual_height + text_size[1]) // 2

            # Ensure text position is not negative
            text_x = max(0, text_x)
            text_y = max(text_size[1], text_y)

            # Draw text on the patch
            cv2.putText(patch, displayed_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)

            # Rotate text if needed
            if rotate_text_180:
                patch = cv2.rotate(patch, cv2.ROTATE_180)

            # Overlay patch onto the image - use exception handling to ensure size match
            try:
                projected[y1:y2, x1:x2] = patch
            except ValueError as e:
                # If sizes don't match, try resizing the patch
                if y2 - y1 > 0 and x2 - x1 > 0:
                    # Resize patch to match target area
                    resized_patch = cv2.resize(patch, (x2 - x1, y2 - y1))
                    projected[y1:y2, x1:x2] = resized_patch
                else:
                    print(f"Skipping keypoint {keypoint_count} due to invalid dimensions")

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

        candidate_keypoints, projected_img = self.get_keypoints(base_rgb, points, masks, rotate_text_180=True, visualize_dino_features=True, vis_mask_features=True, visualize_tsne=False)
        print("Candidate Keypoints:", candidate_keypoints)

        if visualize_projection:
            cv2.imshow('Projected Image', projected_img)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
            projected_img = cv2.rotate(projected_img, cv2.ROTATE_180)
            img_pth = os.path.join(self.save_dir, "scene_img.png")
            cv2.imwrite(img_pth, projected_img)
            # cv2.imwrite('/home/zhuoli/xtrainer_clover/configs/scene_img.png', projected_img)

        # save keypoints as metadata
        candidate_keypoints = candidate_keypoints.tolist()
        keypoints_pth = os.path.join(self.save_dir, "keypoints.json")
        metadata = {
            'keypoint_positions': candidate_keypoints,  # Ensure numpy arrays are converted
            'num_keypoints': len(candidate_keypoints)
        }

        with open(keypoints_pth, 'w') as f:
            json.dump(metadata, f, indent=4)

        return candidate_keypoints

if __name__ == "__main__":
    keypoint_config = get_config(config_path="/home/zhuoli/xtrainer_clover/configs/keypoint_config.yaml")
    keypoint_proposer = KeypointProposer(keypoint_config['keypoint_proposer'])
    keypoints = keypoint_proposer.run(visualize_projection=True)
    print("Keypoints:", keypoints[0])

