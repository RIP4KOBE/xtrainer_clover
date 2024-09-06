import time
import h5py
import os
import glob
import cv2
import pickle
import numpy as np
from scripts.function_util import save_videos, mk_dir
from pathlib import Path
import tyro
from dataclasses import dataclass
#
# """
# For each timestep:
# observations
# - images
#     - each_cam_name     (480, 640, 3) 'uint8'
# - qpos                  (14,)         'float64'
# - qvel                  (14,)         'float64'
#
# action                  (14,)         'float64'
# """
#

@dataclass
class Args:
    dataset_name: str = "tidying_up_bowls_0904"
    task_name: str = "dobot_task"
    MIRROR_STATE_MULTIPLY: list = (1, 1, 1, 1, 1, 1, 1)
    MIRROR_BASE_MULTIPLY: tuple = (1, 1)


def deal_data(pos_list, top_list, left_list, right_list):
    if len(pos_list) < len(top_list):
        for i in range(len(top_list)):
            file_name = top_list[i].split("/")[-1].split(".")[0] + ".pkl"
            if not os.path.exists(os.path.dirname(pos_list[0])+f"/{file_name}"):
                print(top_list[i])
                os.remove(top_list[i])
                os.remove(left_list[i])
                os.remove(right_list[i])
                top_list.remove(top_list[i])
                left_list.remove(left_list[i])
                right_list.remove(right_list[i])
    elif len(pos_list) > len(top_list):
        for i in range(len(pos_list)):
            # file_name = pos_list[i].split("/")[-1].split(".")[0] + ".npy"
            file_name = pos_list[i].split("/")[-1].split(".")[0] + ".jpg"
            if not os.path.exists(os.path.dirname(pos_list[0])+f"/{file_name}"):
                print(pos_list[i])
                os.remove(pos_list[i])
                pos_list.remove(pos_list[i])
    return pos_list, top_list, left_list, right_list


def load_data(one_dataset_dir):
    camera_names = ['top', 'left_wrist', 'right_wrist']
    print(camera_names)

    data_pose_list = glob.glob(one_dataset_dir + 'observation/*.pkl')
    # images_top_list = glob.glob(one_dataset_dir + 'topImg/*.npy')
    # images_left_list = glob.glob(one_dataset_dir + 'leftImg/*.npy')
    # images_right_list = glob.glob(one_dataset_dir + 'rightImg/*.npy')
    images_top_list = glob.glob(one_dataset_dir + 'topImg/*.jpg')
    images_left_list = glob.glob(one_dataset_dir + 'leftImg/*.jpg')
    images_right_list = glob.glob(one_dataset_dir + 'rightImg/*.jpg')
    data_pose_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_top_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_left_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_right_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    print(images_right_list)

    data_pose_list, images_top_list, images_left_list, images_right_list = (
        deal_data(data_pose_list, images_top_list, images_left_list, images_right_list))

    is_sim = False
    qpos = []
    qvel = []
    action = []
    base_action = None
    image_dict = dict()
    image_li = [[], [], []]
    for cam_name in camera_names:
        image_dict[f'{cam_name}'] = []
    for i in range(len(data_pose_list)):
        with open(data_pose_list[i], "rb") as f:
            data_single = pickle.load(f)
            qpos.append(data_single['joint_positions'])
            qvel.append(data_single['joint_velocities'])
            action.append(data_single['control'])
            # image_top = cv2.imdecode(np.asarray(np.load(images_top_list[i]), dtype="uint8"), cv2.IMREAD_COLOR)
            # image_left = cv2.imdecode(np.asarray(np.load(images_left_list[i]), dtype="uint8"), cv2.IMREAD_COLOR)
            # image_right = cv2.imdecode(np.asarray(np.load(images_right_list[i]), dtype="uint8"), cv2.IMREAD_COLOR)
            image_top = cv2.imread(images_top_list[i])
            image_left = cv2.imread(images_left_list[i])
            image_right = cv2.imread(images_right_list[i])
            # cv2.imshow("0", image_right)
            # cv2.waitKey(1)
            image_li[0].append(image_top)
            image_li[1].append(image_left)
            image_li[2].append(image_right)
    image_dict['top'] = np.array(image_li[0])
    image_dict['left_wrist'] = np.array(image_li[1])
    image_dict['right_wrist'] = np.array(image_li[2])
    return np.array(qpos), np.array(qvel), np.array(action), base_action, image_dict, is_sim


def main(args):

    # 1. creating output directory
    # root_dir = str(Path(__file__).parent.parent.parent / "datasets/")
    root_dir = "/media/zhuoli/Elements SE/Tem/xtrainer/datasets"
    dataset_dir = root_dir + "/" + args.dataset_name + "/collect_data/"
    mk_dir(dataset_dir)
    output_video_dir = root_dir + "/" + args.dataset_name + "/output_videos/"
    mk_dir(output_video_dir)
    output_train_data = root_dir + "/" + args.dataset_name + "/train_data/"
    mk_dir(output_train_data)

    all_data_dir = os.listdir(dataset_dir)
    print(all_data_dir)
    all_data_dir.sort(key=lambda x: int(x))
    print(all_data_dir)
    MIRROR_STATE_MULTIPLY = np.array(args.MIRROR_STATE_MULTIPLY)



    for idx in range(len(all_data_dir)):

        # 2. loading and processing data
        print("dealing with : ", idx)
        one_data_dir = dataset_dir+all_data_dir[idx]+"/"
        print(one_data_dir)

        qpos, qvel, action, base_action, image_dict, is_sim = load_data(one_data_dir)
        qpos = np.concatenate([qpos[:, :7] * MIRROR_STATE_MULTIPLY, qpos[:, 7:] * MIRROR_STATE_MULTIPLY],
                              axis=1) # MIRROR_STATE_MULTIPLY variables is used for data augmentation and symmetry
        # handling
        qvel = np.concatenate([qvel[:, :7] * MIRROR_STATE_MULTIPLY, qvel[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
        action = np.concatenate([action[:, :7] * MIRROR_STATE_MULTIPLY, action[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
        if base_action is not None:
            base_action = base_action * args.MIRROR_BASE_MULTIPLY

        if 'left_wrist' in image_dict.keys():
            image_dict['left_wrist'], image_dict['right_wrist'] = \
                image_dict['left_wrist'], image_dict['right_wrist']
        elif 'cam_left_wrist' in image_dict.keys():
            image_dict['cam_left_wrist'], image_dict['cam_right_wrist'] = \
                image_dict['cam_left_wrist'][:, :, ::-1], image_dict['cam_right_wrist'][:, :, ::-1]
        else:
            raise Exception('No left_wrist or cam_left_wrist in image_dict')

        if 'top' in image_dict.keys():
            image_dict['top'] = image_dict['top']
        elif 'cam_high' in image_dict.keys():
            image_dict['cam_high'] = image_dict['cam_high'][:, :, ::-1]
        else:
            raise Exception('No top or cam_high in image_dict')


        # 3.saving data to dict
        data_dict = {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
            '/base_action': base_action,
        } if base_action is not None else {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
        }
        for cam_name in image_dict.keys():
            data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]
        max_timesteps = len(qpos)

        COMPRESS = True

        # 4.compress images if needed
        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
            compressed_len = []
            for cam_name in image_dict.keys():
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image,
                                                         encode_param)  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in image_dict.keys():
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')

        # 5.creating hdf5 dataset for training
        t0 = time.time()
        dataset_path = os.path.join(output_train_data, f'episode_init_{idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = is_sim
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in image_dict.keys():
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                             chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))
            if base_action is not None:
                base_action = root.create_dataset('base_action', (max_timesteps, 2))

            # data saving
            for name, array in data_dict.items():
                root[name][...] = array

            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(image_dict.keys()), max_timesteps))
                root['/compress_len'][...] = compressed_len

        print(f'Saving {dataset_path}: {time.time() - t0:.1f} secs\n')

        # if idx in [0, 4, 8, 23, 33]:
        save_videos(image_dict, 0.02, video_path=os.path.join(output_video_dir + f'{all_data_dir[idx]}_video.mp4'))


if __name__ == "__main__":
    main(tyro.cli(Args))