### Task parameters
import pathlib
import os

DATA_DIR = './data'
TASK_CONFIGS = {
    # dobot move cube new
        'move_cube_new': {
        'dataset_dir': DATA_DIR + '/move_cube_new',
        'episode_len': 900,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    #  dobot move cube cotrain2
    'move_cube_new_cotrain2': {
        'dataset_dir': DATA_DIR + '/move_cube_new',
        'train_ratio': 0.9,  # ratio of train data from the first dataset_dir
        'episode_len': 900,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    # dobot cleandisk
        'clean_disk': {
        'dataset_dir': DATA_DIR + '/clean_disk',
        'episode_len': 1000,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },


    # dobot floder closh
    'floder_closh': {
        'dataset_dir': DATA_DIR + '/floder_closh',
        'episode_len': 2000,  # 1100,  # 900,
        'train_ratio': 0.9,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'floder_closh_cotrain': {
        'dataset_dir': [
            DATA_DIR + '/floader_closh',
            DATA_DIR + '/clean_disk5',
        ],  # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/floder_closh',
        ],
        'sample_weights': [5, 5],
        'train_ratio': 0.9,  # ratio of train data from the first dataset_dir
        'episode_len': 2000,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

}

###  fixed constants
DT = 0.02
FPS = 50




