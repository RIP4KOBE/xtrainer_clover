def reward_fn(trajectory, last_action):
    device = trajectory.device
    trajectory = self.process_trajectory(trajectory, last_action, device=device)
    right_trajectory = trajectory[:, :, RIGHT_ARM_6D_INDICES]

    right_ee_position = right_trajectory[:, :, :3]
    bottle_position = keypoints[1]
    safe_distance = 0.15
    scores = np.zeros(self.sampling_batch_size)

    for i in range(self.sampling_batch_size):
        distances = np.linalg.norm(right_ee_position[i] - bottle_position, axis=1)
        scores[i] = -np.mean((distances - safe_distance) ** 2)

    scores = -torch.as_tensor(scores, device=device)
    return scores, {}