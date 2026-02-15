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

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .base_backend import ArticulationHandle, BaseSimBackend


@dataclass
class IsaacSimLabBackendCfg:
    headless: bool = True
    enable_cameras: bool = False
    device: str = "cpu"
    physics_dt: float = 1.0 / 60.0
    render_interval: int = 1
    lite: bool = False
    experience: str = ""
    kit_args: str = ""


class IsaacSimLabBackend(BaseSimBackend):
    """Isaac Sim + Isaac Lab backend using isaaclab.app.AppLauncher and isaaclab.sim.SimulationContext."""

    def __init__(self, cfg: IsaacSimLabBackendCfg):
        self.cfg = cfg
        self._app_launcher: Any | None = None
        self._simulation_app: Any | None = None
        self._sim: Any | None = None
        self._sim_utils: Any | None = None
        self._robots: list[ArticulationHandle] = []
        self._sim_time_s: float = 0.0

    @property
    def sim(self) -> Any:
        if self._sim is None:
            raise RuntimeError("Simulation is not launched. Call launch() first.")
        return self._sim

    def launch(self) -> None:
        # NOTE: We intentionally do all Isaac Sim/Isaac Lab imports here (after launching the app),
        # since many modules require `carb`/`omni` to be available.
        from isaaclab.app import AppLauncher

        launcher_args = {
            "headless": bool(self.cfg.headless),
            "enable_cameras": bool(self.cfg.enable_cameras),
            "device": str(self.cfg.device),
            "experience": str(self.cfg.experience),
            "kit_args": str(self.cfg.kit_args),
        }
        self._app_launcher = AppLauncher(launcher_args)
        self._simulation_app = self._app_launcher.app

        import isaaclab.sim as sim_utils

        self._sim_utils = sim_utils
        render_interval = int(self.cfg.render_interval)
        if self.cfg.lite and render_interval <= 1:
            # Render less frequently than physics while keeping a GUI window.
            render_interval = 8

        render_cfg = sim_utils.RenderCfg()
        if self.cfg.lite:
            # Turn off expensive RTX features; keep the viewport usable while prioritizing speed.
            render_cfg.antialiasing_mode = "Off"
            render_cfg.enable_shadows = False
            render_cfg.enable_translucency = False
            render_cfg.enable_reflections = False
            render_cfg.enable_global_illumination = False
            render_cfg.enable_ambient_occlusion = False
            render_cfg.enable_dlssg = False
            render_cfg.enable_dl_denoiser = False
            render_cfg.samples_per_pixel = 1
            render_cfg.dlss_mode = 0

        self._sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=float(self.cfg.physics_dt),
                device=str(self.cfg.device),
                render_interval=render_interval,
                render=render_cfg,
            )
        )

        # Make sure we are in a good state before assets are used.
        self._sim_time_s = 0.0

    def close(self) -> None:
        if self._simulation_app is not None:
            self._simulation_app.close()
        self._simulation_app = None
        self._app_launcher = None
        self._sim = None
        self._sim_utils = None
        self._robots.clear()
        self._sim_time_s = 0.0

    def reset(self) -> None:
        # Start playing / initialize PhysX views.
        self.sim.reset()
        # Reset spawned robots to their configured defaults.
        for handle in self._robots:
            robot = handle.asset
            # Root + joint state.
            root_state = robot.data.default_root_state.clone()
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
            robot.reset()
        # Resolve joint indices now that PhysX views are available.
        for handle in self._robots:
            if handle.joint_ids is not None:
                continue
            joint_ids, joint_names = handle.asset.find_joints(list(handle.joint_names), preserve_order=True)
            if len(joint_ids) != len(handle.joint_names):
                raise RuntimeError(
                    f"Controlled joints not found in articulation. Requested: {list(handle.joint_names)}. "
                    f"Available: {handle.asset.joint_names}"
                )
            handle.joint_ids = tuple(joint_ids)
            handle.joint_names = tuple(joint_names)
        self._sim_time_s = 0.0

    def step(self) -> None:
        dt = float(self.sim.get_physics_dt())
        self.sim.step()
        self._sim_time_s += dt
        # Update buffers for all robots.
        for handle in self._robots:
            handle.asset.update(dt)

    def get_sim_time(self) -> float:
        return float(self._sim_time_s)

    def add_dome_light(
        self,
        *,
        prim_path: str = "/World/Light",
        intensity: float = 3000.0,
        color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    ) -> str:
        if self._sim_utils is None:
            raise RuntimeError("Simulation is not launched. Call launch() before adding lights.")
        cfg = self._sim_utils.DomeLightCfg(intensity=float(intensity), color=tuple(map(float, color)))
        cfg.func(str(prim_path), cfg)
        return str(prim_path)

    def spawn_franka_panda(
        self,
        *,
        prim_path: str,
        robot_pose: Sequence[float] | None = None,
        initial_joint_pos: Sequence[float] | None = None,
        use_nucleus_usd: bool = True,
        urdf_path: str | None = None,
        controlled_joint_names: Sequence[str] | None = None,
    ) -> ArticulationHandle:
        try:
            return self._spawn_franka_panda_impl(
                prim_path=prim_path,
                robot_pose=robot_pose,
                initial_joint_pos=initial_joint_pos,
                use_nucleus_usd=use_nucleus_usd,
                urdf_path=urdf_path,
                controlled_joint_names=controlled_joint_names,
            )
        except SystemExit as exc:
            # Isaac Sim / Kit sometimes uses `sys.exit()` internally on configuration/extension failures.
            raise RuntimeError(
                "Isaac Sim terminated the python process during robot spawn (SystemExit). "
                "Check the Isaac Sim log under `~/.nvidia-omniverse/logs/Kit/Isaac-Sim/` or "
                "`.../omni/logs/Kit/Isaac-Sim/` for details. "
                "If Nucleus assets are unavailable, try `use_nucleus_usd=False` and provide `urdf_path`."
            ) from exc

    def _spawn_franka_panda_impl(
        self,
        *,
        prim_path: str,
        robot_pose: Sequence[float] | None = None,
        initial_joint_pos: Sequence[float] | None = None,
        use_nucleus_usd: bool = True,
        urdf_path: str | None = None,
        controlled_joint_names: Sequence[str] | None = None,
    ) -> ArticulationHandle:
        if self._sim_utils is None:
            raise RuntimeError("Simulation is not launched. Call launch() before spawning robots.")

        # Resolve controlled joints (STORM expects 7-DoF Panda arm by default).
        if controlled_joint_names is None:
            controlled_joint_names = [f"panda_joint{i}" for i in range(1, 8)]

        # Convert pose.
        init_pos = None
        init_rot_wxyz = None
        if robot_pose is not None:
            if len(robot_pose) != 7:
                raise ValueError("robot_pose must be [x, y, z, qx, qy, qz, qw].")
            init_pos = (float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2]))
            init_rot_wxyz = (float(robot_pose[6]), float(robot_pose[3]), float(robot_pose[4]), float(robot_pose[5]))

        # Spawn config
        robot_cfg = None
        if use_nucleus_usd and urdf_path is None:
            try:
                from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

                robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=str(prim_path))
                if init_pos is not None:
                    robot_cfg.init_state.pos = init_pos
                if init_rot_wxyz is not None:
                    robot_cfg.init_state.rot = init_rot_wxyz
                if initial_joint_pos is not None:
                    if len(initial_joint_pos) != 7:
                        raise ValueError("initial_joint_pos must have 7 elements for Panda arm joints.")
                    for i, name in enumerate([f"panda_joint{k}" for k in range(1, 8)]):
                        robot_cfg.init_state.joint_pos[name] = float(initial_joint_pos[i])
            except Exception as exc:
                raise RuntimeError(
                    "Failed to spawn Franka Panda from Isaac Lab Nucleus USD. "
                    "If your Nucleus assets are not available, re-run with use_nucleus_usd=False and provide urdf_path."
                ) from exc

        if robot_cfg is None:
            # URDF-based spawn (offline, local file).
            if urdf_path is None:
                raise ValueError("urdf_path must be provided when use_nucleus_usd=False.")

            import os

            urdf_path_abs = os.path.abspath(str(urdf_path))
            if not os.path.isfile(urdf_path_abs):
                raise FileNotFoundError(f"URDF not found: {urdf_path_abs}")

            # Isaac Lab imports (require launched app)
            import isaaclab.sim as sim_utils
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab.assets import Articulation, ArticulationCfg

            # Configure PD gains to match STORM's original Isaac Gym settings.
            actuators = {
                "panda_shoulder": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[1-4]"],
                    effort_limit=87.0,
                    velocity_limit=2.175,
                    stiffness=400.0,
                    damping=40.0,
                ),
                "panda_forearm": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint5"],
                    effort_limit=12.0,
                    velocity_limit=2.61,
                    stiffness=400.0,
                    damping=40.0,
                ),
                "panda_wrist": ImplicitActuatorCfg(
                    joint_names_expr=["panda_joint[6-7]"],
                    effort_limit=12.0,
                    velocity_limit=2.61,
                    stiffness=100.0,
                    damping=5.0,
                ),
            }

            init_joint_pos = {".*": 0.0}
            if initial_joint_pos is not None:
                if len(initial_joint_pos) != 7:
                    raise ValueError("initial_joint_pos must have 7 elements for Panda arm joints.")
                init_joint_pos = {f"panda_joint{i}": float(initial_joint_pos[i - 1]) for i in range(1, 8)}

            init_state = ArticulationCfg.InitialStateCfg(joint_pos=init_joint_pos)
            if init_pos is not None:
                init_state.pos = init_pos
            if init_rot_wxyz is not None:
                init_state.rot = init_rot_wxyz

            robot_cfg = ArticulationCfg(
                prim_path=str(prim_path),
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=urdf_path_abs,
                    fix_base=True,
                    # Keep collisions reasonable and make the asset usable for control.
                    self_collision=True,
                    merge_fixed_joints=True,
                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                        # Provide default values to satisfy config validation; implicit actuators will
                        # set the actual stiffness/damping used for control in the simulation.
                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
                    ),
                ),
                init_state=init_state,
                actuators=actuators,
                soft_joint_pos_limit_factor=1.0,
            )

        # Create the asset instance.
        from isaaclab.assets import Articulation

        robot = Articulation(robot_cfg)
        # Defer resolving joint indices until after reset() when PhysX views are initialized.
        handle = ArticulationHandle(asset=robot, joint_names=tuple(controlled_joint_names))
        self._robots.append(handle)
        return handle

    def _require_joint_ids(self, handle: ArticulationHandle) -> tuple[int, ...]:
        if handle.joint_ids is None:
            raise RuntimeError("ArticulationHandle is not initialized. Call reset() after spawning robots.")
        return handle.joint_ids

    def get_articulation_joint_state(self, handle: ArticulationHandle) -> tuple[np.ndarray, np.ndarray]:
        robot = handle.asset
        joint_ids = self._require_joint_ids(handle)
        # Shapes: (num_envs, num_joints)
        q = robot.data.joint_pos[:, joint_ids].detach().cpu().numpy().squeeze(0)
        qd = robot.data.joint_vel[:, joint_ids].detach().cpu().numpy().squeeze(0)
        return q, qd

    def set_articulation_joint_position_targets(self, handle: ArticulationHandle, q_des: Sequence[float]) -> None:
        robot = handle.asset
        joint_ids = self._require_joint_ids(handle)
        import torch

        q_des = np.asarray(q_des, dtype=np.float32).reshape(1, -1)
        if q_des.shape[1] != handle.dof:
            raise ValueError(f"Expected q_des with {handle.dof} elements, got {q_des.shape[1]}.")
        target = torch.from_numpy(q_des).to(device=robot.device, dtype=robot.data.joint_pos_target.dtype)
        robot.set_joint_position_target(target, joint_ids=list(joint_ids))
        robot.write_data_to_sim()

    def set_articulation_joint_efforts(self, handle: ArticulationHandle, tau: Sequence[float]) -> None:
        robot = handle.asset
        joint_ids = self._require_joint_ids(handle)
        import torch

        tau = np.asarray(tau, dtype=np.float32).reshape(1, -1)
        if tau.shape[1] != handle.dof:
            raise ValueError(f"Expected tau with {handle.dof} elements, got {tau.shape[1]}.")
        target = torch.from_numpy(tau).to(device=robot.device, dtype=robot.data.joint_effort_target.dtype)
        robot.set_joint_effort_target(target, joint_ids=list(joint_ids))
        robot.write_data_to_sim()

    def write_articulation_joint_state(
        self, handle: ArticulationHandle, q: Sequence[float], qd: Sequence[float] | None = None
    ) -> None:
        robot = handle.asset
        joint_ids = self._require_joint_ids(handle)
        import torch

        q_t = torch.as_tensor(
            np.asarray(q, dtype=np.float32).reshape(1, -1),
            device=robot.device,
            dtype=robot.data.joint_pos.dtype,
        )
        if q_t.shape[1] != handle.dof:
            raise ValueError(f"Expected q with {handle.dof} elements, got {q_t.shape[1]}.")
        if qd is None:
            qd_t = torch.zeros_like(q_t)
        else:
            qd_t = torch.as_tensor(
                np.asarray(qd, dtype=np.float32).reshape(1, -1),
                device=robot.device,
                dtype=robot.data.joint_vel.dtype,
            )
        robot.write_joint_state_to_sim(q_t, qd_t, joint_ids=list(joint_ids))
        robot.reset()
