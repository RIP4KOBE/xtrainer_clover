# BiDexDiffuser: Bimanual Dexterous Manipulation Skill Learning with Diffusion Policy

<br>

## Overview

This repo contains code and instructions to support the following use cases:
- Collecting Bimanual Demonstration Data
- Training and Evaluating BiDexDiffuser Policy
- Deploying BiDexDiffuser Policies on Hardware

## Installation

```
conda create -n bidexdiffuser python=3.9
conda activate bidexdiffuser
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r ./requirements.txt
```

## Collecting Demonstration Data


This repo supports Meta Quest 2 as the teleoperation device. To start, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader/blob/main/oculus_reader/reader.py) by following instructions in the link.

We use [ZMQ](https://zeromq.org/) to handle communication between hardwares.
Our data collection code is structured in the following way (credit to [gello_software](https://github.com/wuphilipp/gello_software) for the clean and modular template):

Example usage (note that `launch_node.py` and `run_env.py` should be run simultaneously in two separate windows):

```
# to collect data with two Franka arms and Softhand at 10Hz
python launch_nodes.py --robot bimanual_franka --hand_type softhand
python run_env.py --agent quest_hand --no-show-camera-view --hz 10 --save_data

# to collect data with two Franka arms and Softhand hands at 10Hz, showing camera view during data collection
python launch_nodes.py --robot bimanual_franka --hand_type softhand
python run_env.py --agent quest_hand --hz 10 --save_data
```
Node processes can be cleaned up by running `pkill -9 -f launch_nodes.py`.

## Training and Evaluating Diffusion Policies

[//]: # (### Download Datasets)

[//]: # ([[Data Files]&#40;https://berkeley.app.box.com/s/379cf57zqm1akvr00vdcloxqxi3ucb9g?sortColumn=name&sortDirection=ASC&#41;])

[//]: # ()
[//]: # (The linked data folder contains datasets for the four tasks featured in [the HATO paper]&#40;&#41;: `banana`, `stack`, `pour`, and `steak`.)

[//]: # ()
[//]: # (Full dataset files can be unzipped using the `unzip` command.)

[//]: # (Note that the `pour` and `steak` datasets are split into part files because of the large file size. Before unzipping, the part files should be first concatenated back into single files using the following commands:)

[//]: # (```)

[//]: # (cat data_pour_part_0* > data_pour.zip)

[//]: # (cat data_steak_part_0* > data_steak.zip)

[//]: # (```)

[//]: # ()
[//]: # (Scripts to download and concatenate the datasets can be found in `workflow/download_dataset.sh`.)

### Run Training

1. Run `python workflow/split_data.py --base_path Traj_Folder_Path --output_path Output_Folder_Path --data_name Data_Name --num_trajs N1 N2 N3` to split the data into train and validation sets. Number of trajectories used can be specified via the `num_trajs` argument.
2. Run `python ./learning/dp/pipeline.py --data_path Split_Folder_Path/Data_Name --model_save_path Model_Path` to train the model, where
    - `--data_path` is the splitted trajectory folder, which is the output_path + data_name in step 1. (data_name should not include suffix like `_train` or `_train_10`)
    - `--model_save_path` is the path to save the model

Important Training Arguments
1. `--batch_size` : the batch size for training.
2. `--num_epochs` : the number of epochs for training.
3. `--representation_type`: the data representation type for the model. Format: `repr1--repr2--...`. Repr can be `eef`, `img`, `depth`, `touch`, `pos`, `hand_pos`
4. `--camera_indices`: the camera indices to use for the image data modality. Format: `01`,`012`,`02`, etc.
5. `--train_suffix`: the suffix for the training folder. This is useful when you want to train the model on different data splits and should be used with the `--num_trajs` arg of `split_data.py`. Format: `_10`, `_50`, etc.
6. `--load_img`: whether to load all the images into memory. If set to `True`, the training will be faster but will consume more memory.
7. `--use_memmap_cache`: whether to use memmap cache for the images. If set to `True`, it will create a memmap file in the training folder to accelerate the data loading.
8. `--use_wandb`: whether to use wandb for logging.
9. `--wandb_project_name`: the wandb project name.
10. `--wandb_entity_name`: the wandb entity name.
11. `--load_path`: the path to load the model. If set, the model will be loaded from the path and continue training. This should be the path of non-ema model.


### Run Evaluation

Run `python ./eval_dir.py --eval_dir Traj_Folder_Path --ckpt_path Model_Path_1 Model_Path_2` to evaluate multiple models on all trajectories in the folder.


## Deploying Policies on Hardware

A set of useful bash scripts can be generated using the following command:

```python workflow/gen_deploy_scripts.py -c [ckpt_folder]```

where `ckpt_folder` is a path that contains one or more checkpoints resulted from the training above. The generated bash scripts provide the following functionalities.

- To run policy deployment with asynchronous setup (see Section IV-D in paper for more details), first run `*_inference.sh` to launch the server, then run `*_node.sh` && `*_env_jit.sh` in separate terminal windows to deploy the robot with inference server.

- To run policy deployment without asynchronous setup, run `*_node.sh` && `*_env.sh` in separate terminal windows.

- To run policy evaluation for individual checkpoints, run `*_test.sh`. Note that path to a folder containing trajectories for evaluation needs to be specified in addition to the model checkpoint path.

- To run open-loop policy test, run `*_openloop.sh`.  Note that path to a demonstration data trajectory needs to be specified in addition to the model checkpoint path.


## Acknowledgement

This project was developed with help from the following codebases.

- [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- [gello_software](https://github.com/wuphilipp/gello_software/tree/main)
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/universal_robots_ur5e/ur5e.xml)
- [oculus_reader](https://github.com/rail-berkeley/oculus_reader)
- [hato](https://github.com/ToruOwO/hato)

## The Team
BiDexDiffuser is developed and maintained by the [CLOVER Lab (Collaborative and Versatile Robots Laboratory)](https://feichenlab.com/), CUHK.

[//]: # (## Related repository: Rofunc)

[//]: # ()
[//]: # (We also have a python package robot learning from demonstration and robot manipulation &#40;**Rofunc**&#41;. )

[//]: # ()
[//]: # (> **Repository address: https://github.com/Skylark0924/Rofunc**)

[//]: # ()
[//]: # ([![Release]&#40;https://img.shields.io/github/v/release/Skylark0924/Rofunc&#41;]&#40;https://pypi.org/project/rofunc/&#41;)

[//]: # (![License]&#40;https://img.shields.io/github/license/Skylark0924/Rofunc?color=blue&#41;)

[//]: # (![]&#40;https://img.shields.io/github/downloads/skylark0924/Rofunc/total&#41;)

[//]: # ([![]&#40;https://img.shields.io/github/issues-closed-raw/Skylark0924/Rofunc?color=brightgreen&#41;]&#40;https://github.com/Skylark0924/Rofunc/issues?q=is%3Aissue+is%3Aclosed&#41;)

[//]: # ([![]&#40;https://img.shields.io/github/issues-raw/Skylark0924/Rofunc?color=orange&#41;]&#40;https://github.com/Skylark0924/Rofunc/issues?q=is%3Aopen+is%3Aissue&#41;)

[//]: # ([![Documentation Status]&#40;https://readthedocs.org/projects/rofunc/badge/?version=latest&#41;]&#40;https://rofunc.readthedocs.io/en/latest/?badge=latest&#41;)

[//]: # ([![Build Status]&#40;https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2FSkylark0924%2FRofunc%2Fbadge%3Fref%3Dmain&style=flat&#41;]&#40;https://actions-badge.atrox.dev/Skylark0924/Rofunc/goto?ref=main&#41;)

[//]: # (![]&#40;img/pipeline.png&#41;)
