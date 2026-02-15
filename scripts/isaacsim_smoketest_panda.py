#!/usr/bin/env python
#
# MIT License
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

"""Smoke-test Isaac Sim + Isaac Lab by spawning Franka Panda and stepping physics.

Usage:
  python scripts/isaacsim_smoketest_panda.py --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script without requiring `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from storm_kit.sim.backends.isaacsim_lab_backend import IsaacSimLabBackend, IsaacSimLabBackendCfg
from storm_kit.util_file import get_assets_path, join_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Isaac Sim smoketest: spawn Panda, step sim, print q/qd.")
    parser.add_argument("--headless", action="store_true", default=False, help="Run Isaac Sim headless")
    parser.add_argument(
        "--lite",
        action="store_true",
        default=False,
        help="Reduce rendering cost (disable heavy RTX features and render less often).",
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of physics steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Physics dt (seconds)")
    parser.add_argument("--device", type=str, default="cpu", help='Simulation device: "cpu", "cuda", "cuda:0", ...')
    parser.add_argument(
        "--dome-light-intensity",
        type=float,
        default=3000.0,
        help="DomeLight intensity.",
    )
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

    backend = IsaacSimLabBackend(
        IsaacSimLabBackendCfg(
            headless=bool(args.headless),
            enable_cameras=False,
            device=str(args.device),
            physics_dt=float(args.dt),
            lite=bool(args.lite),
        )
    )

    # STORM's bundled 7-DoF Panda URDF (no gripper).
    panda_urdf = join_path(get_assets_path(), "urdf/franka_description/franka_panda_no_gripper.urdf")

    print("[INFO] Launching Isaac Sim...", flush=True)
    backend.launch()
    try:
        backend.add_dome_light(intensity=float(args.dome_light_intensity))
        print("[INFO] Spawning Franka Panda...", flush=True)
        robot = backend.spawn_franka_panda(
            prim_path="/World/Robot",
            robot_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            initial_joint_pos=[0.0, -1.0, 0.0, -2.0, 0.0, 1.57, 0.78],
            use_nucleus_usd=bool(args.use_nucleus_usd),
            urdf_path=panda_urdf,
        )
        print("[INFO] Resetting simulation...", flush=True)
        backend.reset()

        q0, _ = backend.get_articulation_joint_state(robot)
        q_cmd = q0.copy()

        for i in range(int(args.steps)):
            # simple motion to verify commands are applied
            q_cmd[0] = q0[0] + 0.2 * float(np.sin(2.0 * np.pi * 0.5 * backend.get_sim_time()))
            backend.set_articulation_joint_position_targets(robot, q_cmd)
            backend.step()

            if i % 50 == 0 or i == int(args.steps) - 1:
                q, qd = backend.get_articulation_joint_state(robot)
                q_str = " ".join(f"{x:+.3f}" for x in q[: min(7, q.shape[0])])
                qd_str = " ".join(f"{x:+.3f}" for x in qd[: min(7, qd.shape[0])])
                print(f"[{i:04d}] t={backend.get_sim_time():.3f} q={q_str} qd={qd_str}", flush=True)
    except Exception:  # noqa: BLE001
        # Print the python traceback before closing the SimulationApp (which can terminate the process).
        import traceback

        traceback.print_exc()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
