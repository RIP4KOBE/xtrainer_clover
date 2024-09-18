## Overview

This repo contains code and instructions to support the use of Xtrainer robot:
- Collecting Demonstration Data
- Training ACT Policy
- Training DP Policy
- Deploying Policies on Hardware

## Installation
- System requirements: Ubuntu 20.04, Python 3.8
- [CUDA](https://developer.nvidia.com/cuda-11-8-0-download-archive), [cuDNN](https://developer.nvidia.com/cudnn-downloads) and [Anaconda](https://www.anaconda.com/download#download-section) is required for GPU acceleration.
- Setup virtual environment and install dependencies:
   ```
   conda create -n xtrainer_clover python==3.8.10
   conda activate xtrainer_clover
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements.txt
   pip install -e third_party/DynamixelSDK/python
   pip install -e ModelTrain/detr
   pip install -e robomimic-r2d2
   sudo apt install libxcb-cursor0
   ```

## Collecting Demonstration Data
- Initialize the robot and camera nodes:
   ```
   python 1_find_port.py # Find the port of the robot
   python scripts > 3_just_buttonA.py # adjust the position between leader robot and follower robot
  python 2_get_offset.py # Get the offset of the robot
  python launch_nodes.py # Launch the robot and camera nodes
   ``` 
- Collect demonstration data for ACT training:
   ```
    python run_control.py --agent_name act
   ```
- Collect demonstration data for DP training:
   ```
    python run_control.py --agent_name dp
   ```

## Training and Deploying Policies

### ACT Training

1. Run `python 4_collect2train_data.py` to process the collected data. Note to change the `dataset_name` and `task_name` in the script.
2. Change the following arguments in `ModelTrain/constants.py`:
    - `DATA_DIR`: the path to the processed data.
    - `TASK_CONFIGS`: the task configuration for the model. 
3. Run `python ModelTrain/model_train.py` to train the model, some important Training Arguments:
    - `--ckpt_dir` : the path to save the model.
    - `--task_name` : the task name for the model.
    - `--batch_size`: the batch size for the model.

### DP Training
1. Run `python ModelTrain/dp/split_data.py` to process the collected data. 
2. Run `python ModelTrain/dp/pipeline.py` to train the model. Some important Training Arguments:
    - `--data_path` is the splitted trajectory folder, which is the output_path + data_name. (data_name should not include suffix like `_train` or `_train_10`)
    - `--model_save_path` is the path to save the model

### Deployment
Run `python experiments/run_inference.py` to deploy the trained ACT or DP models.  Important Arguments:
 - `--agent_name`: the name of the agent. set to `act` for ACT policy and `dp` for DP policy.
 - `--act_ckpt_dir`: the path to the trained ACT model.
 - `--dp_ckpt_dir`: the path to the trained DP model.
 - `--episode_len`: the length of the episode. The user can tailor the length of the action sequence for an autonomous reasoning run by modifying this parameter, e.g., when the
the robot has not finished the target action when the autonomous reasoning is running, the step length can be increased; if the robot has some extra actions after the target action, the step length can be decreased.

## Acknowledgement

This project was developed with help from the following codebases.

- [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- [bidexdiffuser](https://github.com/RIP4KOBE/BiDexDiffuser)

## The Team
BiDexDiffuser is developed and maintained by the [CLOVER Lab (Collaborative and Versatile Robots Laboratory)](https://feichenlab.com/), CUHK.
