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
# DEALINGS IN THE SOFTWARE.#
from __future__ import annotations

import numpy as np

from ..sim.backends.isaacsim_lab_backend import IsaacSimLabBackend, IsaacSimLabBackendCfg


class Gym:
    """Isaac Sim + Isaac Lab implementation of the historical `storm_kit.gym.core.Gym` wrapper.

    This keeps the public surface small: step(), get_sim_time(), dt, and env_list.
    """

    def __init__(
        self,
        sim_params: dict | None = None,
        physics_engine: str = "physx",
        compute_device_id: int = 0,
        graphics_device_id: int = 0,
        num_envs: int = 1,
        headless: bool = False,
        enable_cameras: bool = False,
        lite: bool = False,
        render_interval: int = 1,
        device: str | None = None,
        **kwargs,
    ):
        del physics_engine, graphics_device_id, kwargs
        if sim_params is None:
            sim_params = {}

        physics_dt = float(sim_params.get("dt", 1.0 / 60.0))
        # Derive sim device from STORM's old physx config if not explicitly provided.
        if device is None:
            use_gpu = bool(sim_params.get("physx", {}).get("use_gpu", False))
            device = f"cuda:{compute_device_id}" if use_gpu else "cpu"

        backend_cfg = IsaacSimLabBackendCfg(
            headless=bool(headless),
            enable_cameras=bool(enable_cameras),
            device=str(device),
            physics_dt=physics_dt,
            render_interval=int(render_interval),
            lite=bool(lite),
        )
        self.backend = IsaacSimLabBackend(backend_cfg)
        self.backend.launch()

        self.headless = bool(headless)
        self.dt = float(physics_dt)
        # Isaac Gym had env pointers; we use env indices for compatibility.
        self.env_list = list(range(int(num_envs)))

    def step(self) -> bool:
        self.backend.step()
        return True

    def reset(self) -> None:
        self.backend.reset()

    def close(self) -> None:
        self.backend.close()

    def get_sim_time(self) -> float:
        return self.backend.get_sim_time()

    def add_dome_light(
        self,
        *,
        prim_path: str = "/World/Light",
        intensity: float = 3000.0,
        color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    ) -> str:
        return self.backend.add_dome_light(prim_path=prim_path, intensity=intensity, color=color)


class World:
    """Minimal world helper for spawning simple primitives into Isaac Sim.

    Note: STORM's collision costs use world YAMLs for planning. The prims spawned here are primarily for
    visualization and optional physical interaction.
    """

    def __init__(self, gym_instance: Gym, world_params: dict | None = None, base_prim_path: str = "/World/STORM"):
        self.gym_instance = gym_instance
        self.backend = gym_instance.backend
        self.base_prim_path = str(base_prim_path)
        self._obj_count = 0

        if world_params is None:
            return

        # Spawn spheres and cuboids (static).
        spheres = world_params.get("world_model", {}).get("coll_objs", {}).get("sphere", {})
        for name, spec in spheres.items():
            self.spawn_sphere(
                radius=float(spec["radius"]),
                position=spec["position"],
                name=str(name),
                color=(0.6, 0.6, 0.6),
            )

        cubes = world_params.get("world_model", {}).get("coll_objs", {}).get("cube", {})
        for name, spec in cubes.items():
            self.add_table(dims=spec["dims"], pose=spec["pose"], name=str(name), color=(0.6, 0.6, 0.6))

    def _require_sim_utils(self):
        if self.backend._sim_utils is None:
            raise RuntimeError("World can only be used after Gym/Backend is launched.")
        return self.backend._sim_utils

    def spawn_sphere(
        self,
        *,
        radius: float,
        position: Sequence[float],
        name: str | None = None,
        color: tuple[float, float, float] = (0.6, 0.6, 0.6),
    ) -> str:
        sim_utils = self._require_sim_utils()
        name = f"sphere_{self._obj_count}" if name is None else name
        self._obj_count += 1
        prim_path = f"{self.base_prim_path}/spheres/{name}"

        cfg = sim_utils.SphereCfg(
            radius=float(radius),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(map(float, color))),
        )
        cfg.func(prim_path, cfg, translation=(float(position[0]), float(position[1]), float(position[2])))
        return prim_path

    def add_table(
        self,
        dims: Sequence[float],
        pose: Sequence[float],
        name: str = "table",
        color: tuple[float, float, float] = (0.6, 0.6, 0.6),
    ) -> str:
        sim_utils = self._require_sim_utils()
        prim_path = f"{self.base_prim_path}/cuboids/{name}"
        # Pose is [x, y, z, qx, qy, qz, qw] -> orientation expects (w, x, y, z)
        translation = (float(pose[0]), float(pose[1]), float(pose[2]))
        orientation = (float(pose[6]), float(pose[3]), float(pose[4]), float(pose[5]))

        cfg = sim_utils.CuboidCfg(
            size=(float(dims[0]), float(dims[1]), float(dims[2])),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(map(float, color))),
        )
        cfg.func(prim_path, cfg, translation=translation, orientation=orientation)
        return prim_path
