# Installation manual

> **Note**
> Ubuntu 20.04 is highly recommended, since the initial version of python3 is **python 3.8** (which is necessary for this package).  

- [Installation manual](#installation-manual)
  - [Install Ros Neotic](#install-ros-neotic)
  - [Install CuriGPT](#install-curiGPT)
    - [Download the package](#download-the-package)
    - [Install requirements](#install-requirements)
  - [Configure APIs](#configure-apis)
    - [Qwen API](#qwen-api)
    - [OpenAI API](#openai-api)


## Install Ros Neotic

Please follow the [official documentation](http://wiki.ros.org/noetic/Installation/Ubuntu) for Ros Neotic, and also [configure your ros environment](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment).


## Install CuriGPT

### Download the package
Clone the repository into the `src` folder of a catkin workspace

```
git clone https://github.com/RIP4KOBE/curigpt_ros.git
```

### Install requirements
Create and activate a new virtual environment.

```
cd /path/to/curigpt_ros
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```
Install the Python dependencies within the activated virtual environment.
```
pip install -r requirements.txt
```
Build and source the catkin workspace,
```
catkin build curigpt_ros
source /path/to/catkin_ws/devel/setup.bash
```

## Configure APIs
### Qwen API

1. Copy your Qwen API from this [website](https://dashscope.aliyun.com/)
2. Export it to environmental variables:
```
export DASHSCOPE_API_KEY=sk-bfe12097afed4249a73cbafb4fec3e1c
```
### OpenAI API

1. Copy your OpenAI API from this [website](https://platform.openai.com/account/api-keys)
2. Paste it to `config/config.json`.