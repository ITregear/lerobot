#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("joystick")
@dataclass
class JoystickTeleopConfig(TeleoperatorConfig):
    """Configuration for joystick-based teleoperation."""

    device_index: int = 0  # Joystick device index (0 for first joystick)
    deadzone: float = 0.1  # Deadzone for joystick axes to prevent drift
    step_size: float = 0.05  # Step size for relative movement sensitivity
    axis_mapping: dict[int, str] | None = None  # Mapping of joystick axes to robot joints

    def __post_init__(self):
        if self.id is None:
            self.id = "fsi6x"

        if self.axis_mapping is None:
            # Default mapping for SO101 - can be overridden
            self.axis_mapping = {
                0: "shoulder_pan",
                1: "shoulder_lift",
                2: "elbow_flex",
                3: "wrist_flex",
                4: "wrist_roll",
                5: "gripper",
            }


@TeleoperatorConfig.register_subclass("joystick_ee")
@dataclass
class JoystickEndEffectorTeleopConfig(JoystickTeleopConfig):
    """Configuration for joystick-based end-effector teleoperation."""

    # End-effector axis mapping
    ee_axis_mapping: dict[int, str] | None = None

    # Step sizes for end-effector movement
    ee_step_sizes: dict[str, float] | None = None

    def __post_init__(self):
        super().__post_init__()
        
        if self.ee_axis_mapping is None:
            # Default mapping for end-effector control
            self.ee_axis_mapping = {
                0: "x",      # Left/right movement
                1: "y",      # Forward/backward movement  
                2: "z",      # Up/down movement
                3: "roll",   # End-effector roll (orientation)
                5: "gripper", # Gripper control (keeping existing mapping)
            }

        if self.ee_step_sizes is None:
            # Default step sizes for end-effector movement (in meters)
            self.ee_step_sizes = {
                "x": 0.02,      # 2cm per step
                "y": 0.02,      # 2cm per step
                "z": 0.02,      # 2cm per step
                "roll": 0.1,    # ~5.7 degrees per step
                "gripper": 5.0, # 5% gripper change per step
            }
