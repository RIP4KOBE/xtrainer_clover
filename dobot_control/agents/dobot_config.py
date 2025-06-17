import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
from dobot_control.robots.dynamixel import DynamixelRobot



@dataclass
class DobotRobotConfig:
    joint_ids: Sequence[int]
    append_id: int
    port: str
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Tuple[int, int, int]
    start_joints: Sequence[float]

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(self, start_joints: Optional[np.ndarray] = None) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            append_id=self.append_id,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=self.port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
        )