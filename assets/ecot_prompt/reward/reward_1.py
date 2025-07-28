def reward_fn(trajectory, last_action):
    device = trajectory.device
    trajectory = self.process_trajectory(trajectory, last_action, device=device)
    right_trajectory = trajectory[:, :, RIGHT_ARM_6D_INDICES]

    right_ee_position = right_trajectory[:, :, :3]
    scores = np.zeros(self.sampling_batch_size)

    for i in range(self.sampling_batch_size):
        initial_height = right_ee_position[i, 0, 2]
        final_height = right_ee_position[i, -1, 2]
        rel_height = initial_height - final_height
        scores[i] = -abs(rel_height - 0.05)  #
    scores = -torch.as_tensor(scores, device=device)
    return scores, {}