
# CuriGPT: Interactive Humanoid Manipulation by Multimodal Large Language Models

- [CuriGPT: Interactive Humanoid Manipulation by Multimodal Large Language Models](#rofunc-ros-a-ros-package-for-human-centered-intelligent-interactive-humanoid-robots)
  - [Installation](#installation)
    - [System requirements](#system-requirements)
    - [Installation](#installation-1)
  - [Functions](#functions)
    - [Multimodal Voice Q&A](#multimodal-voice-qa)

[//]: # (  - [Cite]&#40;#cite&#41;)
  - [The Team](#the-team)

[//]: # (  - [Related repository: Rofunc]&#40;#related-repository-rofunc&#41;)
  - [Acknowledge](#acknowledge)


## Installation

### System requirements

The package is only tested by the following configuration.

1. Ubuntu 20.04
2. Ros Neotic

### Installation

**Please refer to the [installation manual](docs/Installation.md).**

## Functions

### Multimodal Voice Q&A

For this function, we implemented a Voice Q&A robot based on multimodal large language models`qwen_vl_chat_v1`. 

[//]: # (![]&#40;img/voiceQA_pipeline.png&#41;)

This whole pipeline can be activated by calling

```
cd /path/to/curigpt_ros
python3 scripts/curigpt.py
```

[//]: # (## Cite)

[//]: # ()
[//]: # (If you use rofunc-ros in a scientific publication, we would appreciate citations to the following paper:)

[//]: # ()
[//]: # (```)

[//]: # (@misc{Rofunc2022,)

[//]: # (      author = {Liu, Junjia and Li, Zhihao and Li, Chenzui and Chen, Fei},)

[//]: # (      title = {Rofunc: The full process python package for robot learning from demonstration},)

[//]: # (      year = {2022},)

[//]: # (      publisher = {GitHub},)

[//]: # (      journal = {GitHub repository},)

[//]: # (      howpublished = {\url{https://github.com/Skylark0924/Rofunc}},)

[//]: # (})

[//]: # (```)

## The Team
CuriGPT is developed and maintained by the [CLOVER Lab (Collaborative and Versatile Robots Laboratory)](https://feichenlab.com/), CUHK.

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



## Acknowledge

We would like to acknowledge the following projects:

1. [wpr_chatgpt](https://github.com/play-with-chatgpt/wpr_chatgpt/)
2. [ros-vosk](https://github.com/alphacep/ros-vosk)
3. [tts-ros1](https://github.com/aws-robotics/tts-ros1)
