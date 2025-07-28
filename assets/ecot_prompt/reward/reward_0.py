def reward_fn(trajectory, last_action):
    device = trajectory.device
    trajectory = self.process_trajectory(trajectory, last_action, device=device)
    left_trajectory = trajectory[:, :, LEFT_ARM_6D_INDICES]

    left_ee_position = left_trajectory[:, :, :3]
    scores = np.zeros(self.sampling_batch_size)

    for i in range(self.sampling_batch_size):
        initial_height = left_ee_position[i, 0, 2]
        final_height = left_ee_position[i, -1, 2]
        scores[i] = final_height - initial_height
    scores = -torch.as_tensor(scores, device=device)
    return scores, {}