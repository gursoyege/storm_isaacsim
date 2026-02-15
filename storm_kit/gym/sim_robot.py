#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np

from ..sim.backends.base_backend import ArticulationHandle, BaseSimBackend


class RobotSim:
    """Robot wrapper matching STORM's historical Isaac Gym interface.

    The Isaac Gym implementation exposed:
      - spawn_robot()
      - get_state()
      - command_robot_position()
      - command_robot()

    This version uses a simulator backend (Isaac Sim + Isaac Lab).
    """

    def __init__(
        self,
        device: str = "cpu",
        gym_instance: Any | None = None,
        sim_instance: Any | None = None,
        asset_root: str = "",
        sim_urdf: str = "",
        asset_options: dict | None = None,
        init_state: Sequence[float] | None = None,
        collision_model: Any | None = None,
        use_nucleus_usd: bool = True,
        controlled_joint_names: Sequence[str] | None = None,
        **kwargs,
    ):
        del device, sim_instance, asset_options, collision_model, kwargs

        # Resolve backend
        backend = None
        if gym_instance is not None and hasattr(gym_instance, "backend"):
            backend = gym_instance.backend
        elif gym_instance is not None and hasattr(gym_instance, "spawn_franka_panda"):
            backend = gym_instance
        if backend is None:
            raise ValueError("RobotSim requires a Gym instance (storm_kit.gym.core.Gym) or a simulation backend.")
        if not isinstance(backend, BaseSimBackend):
            # Duck-typing is fine, but keep error message actionable.
            raise TypeError(f"gym_instance.backend must implement BaseSimBackend. Got: {type(backend)}")

        self.backend: BaseSimBackend = backend
        self.use_nucleus_usd = bool(use_nucleus_usd)
        self.controlled_joint_names = (
            list(controlled_joint_names) if controlled_joint_names is not None else [f"panda_joint{i}" for i in range(1, 8)]
        )

        self.init_state = np.asarray(init_state, dtype=np.float32) if init_state is not None else None

        self.urdf_path: str | None = None
        if sim_urdf:
            self.urdf_path = os.path.abspath(os.path.join(str(asset_root), str(sim_urdf)))

        self.joint_names: list[str] = []
        self.dof: int | None = None
        self.robot_handle: ArticulationHandle | None = None

    def spawn_robot(
        self,
        env_handle: int,
        robot_pose: Sequence[float],
        robot_asset: Any | None = None,
        coll_id: int = -1,
        init_state: Sequence[float] | None = None,
    ) -> ArticulationHandle:
        del robot_asset, coll_id

        if init_state is None:
            init_state = self.init_state

        prim_path = f"/World/STORM/env_{int(env_handle)}/Robot"
        handle = self.backend.spawn_franka_panda(
            prim_path=prim_path,
            robot_pose=robot_pose,
            initial_joint_pos=None if init_state is None else list(init_state),
            use_nucleus_usd=bool(self.use_nucleus_usd),
            urdf_path=self.urdf_path,
            controlled_joint_names=self.controlled_joint_names,
        )
        self.robot_handle = handle
        self.joint_names = list(handle.joint_names)
        self.dof = handle.dof
        return handle

    def get_state(self, env_handle: int, robot_handle: ArticulationHandle) -> dict:
        del env_handle
        q, qd = self.backend.get_articulation_joint_state(robot_handle)
        joint_state = {
            "name": list(robot_handle.joint_names),
            "position": np.asarray(q, dtype=np.float32).copy(),
            "velocity": np.asarray(qd, dtype=np.float32).copy(),
            "acceleration": np.zeros_like(q, dtype=np.float32),
        }
        return joint_state

    def command_robot(self, tau: Sequence[float], env_handle: int, robot_handle: ArticulationHandle) -> None:
        del env_handle
        self.backend.set_articulation_joint_efforts(robot_handle, tau)

    def command_robot_position(self, q_des: Sequence[float], env_handle: int, robot_handle: ArticulationHandle) -> None:
        del env_handle
        self.backend.set_articulation_joint_position_targets(robot_handle, q_des)

    def set_robot_state(
        self,
        q_des: Sequence[float],
        qd_des: Sequence[float] | None,
        env_handle: int,
        robot_handle: ArticulationHandle,
    ) -> None:
        del env_handle
        self.backend.write_articulation_joint_state(robot_handle, q_des, qd_des)

