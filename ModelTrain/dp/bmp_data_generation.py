import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pathlib
from tqdm import tqdm
import zarr
from ModelTrain.dp.bimanual_motion_prior.replay_buffer import ReplayBuffer
from ModelTrain.dp.utils import ik_solver


def draw_bounding_box(ax, bounds, color='gray', alpha=0.2):
    """Draws a 3D bounding box given workspace bounds."""
    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_min, z_max = bounds['z']

    # 8 corner points
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_min, y_max, z_min], [x_max, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_min, y_max, z_max], [x_max, y_max, z_max]
    ])

    # 12 lines
    edges = [
        (0, 1), (0, 2), (1, 3), (2, 3),  # bottom
        (4, 5), (4, 6), (5, 7), (6, 7),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)  # vertical
    ]

    for start, end in edges:
        ax.plot(*zip(corners[start], corners[end]), color=color, alpha=alpha)

def load_zarr_dataset(data_path):
    """
    Load all top-level arrays from a Zarr dataset directory.

    Parameters:
    - data_path (str): Path to the Zarr dataset (e.g., "real_pusht_20230105/replay_buffer.zarr/data")

    Returns:
    - dataset_dict (dict): Dictionary with each dataset field as a NumPy array
    """
    dataset = zarr.open(data_path, mode='r')
    dataset_dict = {}

    print(f"Reading Zarr dataset from: {data_path}")
    print(f"Found fields: {list(dataset.array_keys())}")

    for field in dataset.array_keys():
        print(f"\nLoading field: '{field}'")
        try:
            array = dataset[field][:]
            print(f"Shape: {array.shape}, Dtype: {array.dtype}")
            dataset_dict[field] = array
        except Exception as e:
            print(f"Failed to load field '{field}': {e}")

    return dataset_dict


class BimanualMotionPriorGenerator:
    def __init__(
        self,
        num_pos_samples=2,
        num_ori_samples_per_axis=1,
        num_traj_per_start=5,
        traj_len=5,
        delta_mag=0.02,
        rot_deg_per_step=5,
        left_workspace_bounds=None,
        right_workspace_bounds=None,
        collision_safety_margin=0.05,
        output_dir='../../datasets/bmp_dataset'
    ):
        # Configurations
        self.num_pos_samples = num_pos_samples
        self.num_ori_samples_per_axis = num_ori_samples_per_axis
        self.num_traj_per_start = num_traj_per_start
        self.traj_len = traj_len
        self.delta_mag = delta_mag
        self.rot_deg_per_step = rot_deg_per_step
        self.collision_safety_margin = collision_safety_margin
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Workspace bounds
        self.left_workspace_bounds = {
            'x': (-0.5, 0.2),
            'y': (-0.6, 0.0),
            'z': (0.2, 0.7)
        }
        self.right_workspace_bounds = {
            'x': (-0.5, 0.2),
            'y': (-0.6, 0.0),
            'z': (0.2, 0.7)
        }

        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=str(self.output_dir / 'replay_buffer.zarr'), mode='a'
        )

    def create_position_grid(self, bounds):
        x = np.linspace(*bounds['x'], self.num_pos_samples)
        y = np.linspace(*bounds['y'], self.num_pos_samples)
        z = np.linspace(*bounds['z'], self.num_pos_samples)
        return np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

    def generate_uniform_orientations(self):
        angles = np.linspace(0, 360, self.num_ori_samples_per_axis, endpoint=False)
        orientations = []
        for roll in angles:
            for pitch in angles:
                for yaw in angles:
                    rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
                    orientations.append(rot.as_quat())
        return orientations

    def fibonacci_sphere(self, samples):
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    def generate_pose_deltas(self):
        directions = self.fibonacci_sphere(self.num_traj_per_start)

        pos_deltas = np.stack([
            direction * self.delta_mag * np.linspace(1, self.traj_len, self.traj_len)[:, None]
            for direction in directions
        ])

        ori_deltas = []
        for _ in range(self.num_traj_per_start):
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            degrees = np.linspace(1, self.traj_len, self.traj_len) * self.rot_deg_per_step
            quats = [R.from_rotvec(np.deg2rad(deg) * axis).as_quat() for deg in degrees]
            ori_deltas.append(quats)

        return pos_deltas, np.array(ori_deltas)

    def is_collision_free(self, left_seq, right_seq):
        for l, r in zip(left_seq, right_seq):
            if np.linalg.norm(l - r) < self.collision_safety_margin:
                return False
        return True

    def dummy_ik_solver(self, pose, is_left=True):
        bounds = self.left_workspace_bounds if is_left else self.right_workspace_bounds
        pos = pose[:3]
        return np.all(pos >= np.array([bounds['x'][0], bounds['y'][0], bounds['z'][0]])) and \
               np.all(pos <= np.array([bounds['x'][1], bounds['y'][1], bounds['z'][1]]))

    def generate(self):
        # Initialization
        left_positions = self.create_position_grid(self.left_workspace_bounds)
        right_positions = self.create_position_grid(self.right_workspace_bounds)
        print("Number of positions - Left: {}, Right: {}".format(len(left_positions), len(right_positions)))

        orientations = self.generate_uniform_orientations()
        print("Number of orientations: {}".format(len(orientations)))

        self.ik_solver = ik_solver

        print("Generating reachable poses...")

        # Compute reachable poses for the left arm with a progress bar
        left_reachables = []
        for pos in tqdm(left_positions, desc="Left IK"):
            for quat in orientations:
                if self.ik_solver(pos, quat)[1]:
                    left_reachables.append(np.concatenate([pos, quat]))

        # Compute reachable poses for the right arm with a progress bar
        right_reachables = []
        for pos in tqdm(right_positions, desc="Right IK"):
            for quat in orientations:
                if self.ik_solver(pos, quat)[1]:
                    right_reachables.append(np.concatenate([pos, quat]))

        print(f"Total reachable poses - Left: {len(left_reachables)}, Right: {len(right_reachables)}")

        # left_reachables = [np.concatenate([pos, quat]) for pos in left_positions for quat in orientations if
        #                    self.ik_solver(pos, quat)[1]]
        # right_reachables = [np.concatenate([pos, quat]) for pos in right_positions for quat in orientations if self.ik_solver(pos, quat)[1]]
        #
        # print(f"Total reachable poses - Left: {len(left_reachables)}, Right: {len(right_reachables)}")

        for l_pose in left_reachables:
            for r_pose in right_reachables:
                l_pos, l_quat = l_pose[:3], l_pose[3:]
                r_pos, r_quat = r_pose[:3], r_pose[3:]

                pos_deltas, ori_deltas = self.generate_pose_deltas()

                for i in range(self.num_traj_per_start):
                    l_traj_pos = l_pos + pos_deltas[i]
                    r_traj_pos = r_pos + pos_deltas[i]

                    l_traj_ori = [R.from_quat(l_quat) * R.from_quat(ori_deltas[i, t]) for t in range(self.traj_len)]
                    r_traj_ori = [R.from_quat(r_quat) * R.from_quat(ori_deltas[i, t]) for t in range(self.traj_len)]

                    l_traj_quat = np.array([rot.as_quat() for rot in l_traj_ori])
                    r_traj_quat = np.array([rot.as_quat() for rot in r_traj_ori])

                    # check if the trajectory is valid
                    # if not all(self.ik_solver(l_traj_pos[t], l_traj_quat[t])[1] for t in range(self.traj_len)):
                    #     continue
                    # if not all(self.ik_solver(r_traj_pos[t], r_traj_quat[t])[1] for t in range(self.traj_len)):
                    #     continue
                    # if not self.is_collision_free(l_traj_pos, r_traj_pos):
                    #     continue

                    obs = np.concatenate([l_pos, l_quat, r_pos, r_quat])
                    obs = obs.reshape(1,-1) .repeat(self.traj_len, 0)

                    # l_delta_pos = l_traj_pos - l_pos
                    # r_delta_pos = r_traj_pos - r_pos
                    #
                    # l_delta_quat = np.array([
                    #     (l_traj_ori[t] * R.from_quat(l_quat).inv()).as_quat()
                    #     for t in range(self.traj_len)
                    # ])
                    # r_delta_quat = np.array([
                    #     (r_traj_ori[t] * R.from_quat(r_quat).inv()).as_quat()
                    #     for t in range(self.traj_len)
                    # ])

                    l_delta_pos =pos_deltas[i]
                    l_delta_quat = ori_deltas[i]
                    r_delta_pos = pos_deltas[i]
                    r_delta_quat = ori_deltas[i]

                    action = np.concatenate([l_delta_pos, l_delta_quat, r_delta_pos, r_delta_quat], axis=1)

                    episode = {
                        'eef_pose': obs.astype(np.float32),
                        'action': action.astype(np.float32)
                    }
                    self.replay_buffer.add_episode(episode, compressors='disk')
                    print(f"Saved episode {self.replay_buffer.n_episodes - 1}")

    def vis_trajs(self,
            downsample=10,
            arrow_length=0.05,
    ):
        """
        Visualize bimanual trajectories with pose arrows and bounding boxes.

        Args:
            downsample (int): Show 1 trajectory every `downsample` episodes
            arrow_length (float): length of orientation arrows
        """

        # Load zarr
        data_path = self.output_dir / 'replay_buffer.zarr' / 'data'
        root = zarr.open(str(data_path), mode='r')
        obs_array = root['eef_pose'][:, :]  # [N, 14]
        action_array = root['action'][:, :]  # [N, T, 14]
        total_episodes = len(obs_array) // self.traj_len

        # Setup figure
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        ax1.set_title("Left Arm - Spherical Trajectories")
        ax2.set_title("Right Arm - Spherical Trajectories")

        for ax in (ax1, ax2):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        # Plot trajectories
        for i in range(0, total_episodes, self.traj_len):
            obs = obs_array[i]
            action = action_array[i:i + self.traj_len]

            # ---- Parse Left Arm ----
            l_start_pos = obs[0:3]
            l_start_quat = obs[3:7]
            l_delta_pos = action[:, :3]
            l_traj_pos = l_start_pos + l_delta_pos

            # ---- Parse Right Arm ----
            r_start_pos = obs[7:10]
            r_start_quat = obs[10:14]
            r_delta_pos = action[:, 7:10]
            r_traj_pos = r_start_pos + r_delta_pos

            # ---- Plot Left ----
            ax1.scatter(*l_start_pos, color='blue', marker='o', s=30)
            ax1.plot(l_traj_pos[:, 0], l_traj_pos[:, 1], l_traj_pos[:, 2], color='blue', alpha=0.5)

            # Draw orientation arrow (forward axis)
            # l_rot = R.from_quat(l_start_quat)
            # l_dir = l_rot.apply([1, 0, 0])  # X axis
            # ax1.quiver(*l_start_pos, *l_dir, length=arrow_length, color='black')

            # ---- Plot Right ----
            ax2.scatter(*r_start_pos, color='red', marker='o', s=30)
            ax2.plot(r_traj_pos[:, 0], r_traj_pos[:, 1], r_traj_pos[:, 2], color='red', alpha=0.5)

            print("processing episode: ", i)

            # r_rot = R.from_quat(r_start_quat)
            # r_dir = r_rot.apply([1, 0, 0])
            # ax2.quiver(*r_start_pos, *r_dir, length=arrow_length, color='black')

        # Draw bounding boxes
        # if left_bounds:
        #     draw_bounding_box(ax1, left_bounds, color='green', alpha=0.4)
        # if right_bounds:
        #     draw_bounding_box(ax2, right_bounds, color='green', alpha=0.4)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    check_dataset = False

    if not check_dataset:
        generator = BimanualMotionPriorGenerator()
        generator.generate()
        generator.vis_trajs()

    else:
        # Load the dataset
        data_path = "../../datasets/bmp_dataset/replay_buffer.zarr/data"
        dataset = load_zarr_dataset(data_path)
        print("Dataset loaded successfully.")
