#
# Copyright (c) 2026.
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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class ArticulationHandle:
    """Opaque handle for an articulated robot in the simulator backend."""

    asset: Any
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...] | None = None

    @property
    def dof(self) -> int:
        return len(self.joint_names)


class BaseSimBackend(ABC):
    """Backend interface for STORM's simulation integration.

    This interface is intentionally minimal: it only covers the functions STORM needs
    for a control loop (step simulation, read joint state, apply joint commands).
    """

    @abstractmethod
    def launch(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def step(self) -> None: ...

    @abstractmethod
    def get_sim_time(self) -> float: ...

    @abstractmethod
    def add_dome_light(
        self,
        *,
        prim_path: str = "/World/Light",
        intensity: float = 3000.0,
        color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    ) -> str: ...

    # --- Robot spawning ---
    @abstractmethod
    def spawn_franka_panda(
        self,
        *,
        prim_path: str,
        robot_pose: Sequence[float] | None = None,
        initial_joint_pos: Sequence[float] | None = None,
        use_nucleus_usd: bool = True,
        urdf_path: str | None = None,
        controlled_joint_names: Sequence[str] | None = None,
    ) -> ArticulationHandle: ...

    # --- Robot I/O ---
    @abstractmethod
    def get_articulation_joint_state(self, handle: ArticulationHandle) -> tuple[np.ndarray, np.ndarray]:
        """Returns (q, qd) as numpy arrays for the controlled joints."""

    @abstractmethod
    def set_articulation_joint_position_targets(self, handle: ArticulationHandle, q_des: Sequence[float]) -> None: ...

    @abstractmethod
    def set_articulation_joint_efforts(self, handle: ArticulationHandle, tau: Sequence[float]) -> None: ...

    @abstractmethod
    def write_articulation_joint_state(
        self, handle: ArticulationHandle, q: Sequence[float], qd: Sequence[float] | None = None
    ) -> None: ...
