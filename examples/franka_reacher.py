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

"""Franka reacher example running STORM MPPI with Isaac Sim + Isaac Lab.

This ports the historical Isaac Gym example to Isaac Sim (isaacsim==4.5.0) using Isaac Lab.

Example:
  python examples/franka_reacher.py --headless --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as a script without requiring `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from storm_kit.gym.core import Gym
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.mpc.task.reacher_task import ReacherTask
from storm_kit.util_file import get_assets_path, get_gym_configs_path, join_path, load_yaml


def main() -> None:
    # Isaac Lab uses multiprocessing in a few utilities; keep behavior consistent with legacy example.
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="STORM Franka reacher using Isaac Sim + Isaac Lab.")
    parser.add_argument("--robot", type=str, default="franka", help="Robot config name under content/configs/gym/")
    parser.add_argument("--world", type=str, default="collision_primitives_3d.yml", help="World YAML under gym configs")
    parser.add_argument("--headless", action="store_true", default=False, help="Run Isaac Sim headless")
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MPPI on CUDA (if available). Use `--no-cuda` to keep the GPU free for rendering.",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        default=False,
        help="Reduce rendering cost (disable heavy RTX features and render less often).",
    )
    parser.add_argument(
        "--dome-light-intensity",
        type=float,
        default=3000.0,
        help="DomeLight intensity.",
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of control steps")
    parser.add_argument("--print-every", type=int, default=20, help="Print joint state every N control steps")
    parser.add_argument(
        "--use-nucleus-usd",
        action="store_true",
        default=True,
        help="Spawn Panda from Isaac Lab Nucleus USD (preferred). If this fails, re-run with --no-use-nucleus-usd.",
    )
    parser.add_argument(
        "--no-use-nucleus-usd",
        dest="use_nucleus_usd",
        action="store_false",
        help="Spawn Panda from STORM URDF via Isaac Lab's URDF converter (offline/local).",
    )
    args = parser.parse_args()

    # --- Simulation setup ---
    sim_cfg = load_yaml(join_path(get_gym_configs_path(), "physx.yml"))
    sim_cfg["headless"] = bool(args.headless)
    sim_cfg["lite"] = bool(args.lite)
    if bool(args.lite):
        sim_cfg["render_interval"] = max(int(sim_cfg.get("render_interval", 1)), 8)
    gym = Gym(**sim_cfg)
    gym.add_dome_light(intensity=float(args.dome_light_intensity))

    # --- Robot spawn ---
    robot_params = load_yaml(join_path(get_gym_configs_path(), f"{args.robot}.yml"))
    sim_params = robot_params["sim_params"]
    sim_params["asset_root"] = get_assets_path()
    sim_params["use_nucleus_usd"] = bool(args.use_nucleus_usd)

    robot_sim = RobotSim(gym_instance=gym, **sim_params)
    env_id = gym.env_list[0]
    robot_handle = robot_sim.spawn_robot(env_id, sim_params["robot_pose"], init_state=sim_params.get("init_state"))

    # Play the sim once the scene is designed.
    gym.reset()

    # --- MPPI controller setup ---
    device = torch.device("cuda", 0) if args.cuda else torch.device("cpu")
    tensor_args = {"device": device, "dtype": torch.float32}

    task_file = f"{args.robot}_reacher.yml"
    robot_file = f"{args.robot}.yml"
    mpc_control = ReacherTask(task_file=task_file, robot_file=robot_file, world_file=args.world, tensor_args=tensor_args)
    control_dt = float(mpc_control.exp_params["control_dt"])

    # --- Goal pose ---
    # STORM's reacher rollouts require an explicit Cartesian goal pose.
    # Use the current EE pose (from the kinematic model) with a small offset so the example runs out of the box.
    current_robot_state = robot_sim.get_state(env_id, robot_handle)
    q0 = torch.as_tensor(current_robot_state["position"], **tensor_args).unsqueeze(0)
    qd0 = torch.as_tensor(current_robot_state["velocity"], **tensor_args).unsqueeze(0)
    ee_link = str(mpc_control.exp_params["model"]["ee_link_name"])
    ee_pos, ee_rot = mpc_control.controller.rollout_fn.dynamics_model.robot_model.compute_forward_kinematics(
        q0, qd0 * 0.0, link_name=ee_link
    )
    goal_offset = torch.as_tensor([0.10, 0.00, 0.10], **tensor_args).unsqueeze(0)
    goal_ee_pos = (ee_pos + goal_offset).squeeze(0).detach().cpu().numpy()
    goal_ee_rot = ee_rot.squeeze(0).detach().cpu().numpy()
    mpc_control.update_params(goal_ee_pos=goal_ee_pos, goal_ee_rot=goal_ee_rot)

    # If the control dt differs from physics dt, keep the command constant for multiple physics steps.
    steps_per_control = max(1, int(round(control_dt / float(gym.dt))))
    effective_control_dt = steps_per_control * float(gym.dt)
    if abs(effective_control_dt - control_dt) > 1.0e-6:
        print(
            f"[WARN] control_dt={control_dt} is not a multiple of physics_dt={gym.dt}. "
            f"Using steps_per_control={steps_per_control} (effective_control_dt={effective_control_dt})."
        , flush=True)

    try:
        for i in range(int(args.steps)):
            t_step = gym.get_sim_time()
            current_robot_state = robot_sim.get_state(env_id, robot_handle)
            command = mpc_control.get_command(t_step, current_robot_state, control_dt=control_dt, WAIT=True)

            q_des = np.asarray(command["position"], dtype=np.float32)
            robot_sim.command_robot_position(q_des, env_id, robot_handle)

            for _ in range(steps_per_control):
                gym.step()

            if int(args.print_every) > 0 and (i % int(args.print_every) == 0):
                q = current_robot_state["position"]
                qd = current_robot_state["velocity"]
                print(f"[{i:04d}] t={t_step:.3f} q[0]={q[0]:+.3f} qd[0]={qd[0]:+.3f}", flush=True)
    except Exception:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        # Ensure child processes are terminated cleanly.
        mpc_control.close()
        gym.close()


if __name__ == "__main__":
    main()
