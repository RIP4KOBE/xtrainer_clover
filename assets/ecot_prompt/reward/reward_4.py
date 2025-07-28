def reward_fn(trajectory, last_action):
    device = trajectory.device
    trajectory = self.process_trajectory(trajectory, last_action, device=device)
    left_trajectory = trajectory[:, :, LEFT_ARM_6D_INDICES]

    left_ee_position = left_trajectory[:, :, :3]
    plate_position = keypoints[18]
    safe_distance = 0.05  # Minimum distance to the bottle
    scores = np.zeros(self.sampling_batch_size)

    for i in range(self.sampling_batch_size):
        final_position = left_ee_position[i, -1, :3]
        distances = np.linalg.norm(final_position - plate_position)
        scores[i] = -np.mean((distances - safe_distance) ** 2)
    scores = -torch.as_tensor(scores, device=device)
    return scores, {}