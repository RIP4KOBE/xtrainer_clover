def reward_fn(trajectory, last_action):
    device = trajectory.device
    trajectory = self.process_trajectory(trajectory, last_action, device=device)
    left_trajectory = trajectory[:, :, LEFT_ARM_6D_INDICES]
    left_ee_rot = left_trajectory[:, :, 3:9]
    scores = np.zeros(self.sampling_batch_size)

    for i in range(self.sampling_batch_size):
        rot_mats = np.stack([sixd_to_rotation_matrix(sixd) for sixd in left_ee_rot[i]])
        initial_rot = rot_mats[0, :, :]
        final_rot = rot_mats[-1, :, :]
        rot_rel = initial_rot.T @ final_rot
        euler = R.from_matrix(rot_rel).as_euler('ZYX', degrees=False)  # (3,)
        yaw, pitch, roll = euler[0], euler[1], euler[2]
        target_yaw = math.radians(30)
        scores[i] = -(
                (yaw - target_yaw) ** 2 +
                (pitch ** 2 + roll ** 2)
        )

    scores = -torch.as_tensor(scores, device=device)
    return scores, {}