## Install Instructions

### System Dependencies (this workspace)
- System Python that can import `isaacsim` (tested with `isaacsim==4.5.0`)
- `IsaacLab` installed into the same system Python (importable as `isaaclab`)

Steps:

1. Verify Isaac Sim + Isaac Lab are importable:

   `python -c "import isaacsim, isaaclab; print('isaacsim', isaacsim.__file__); print('isaaclab', isaaclab.__file__)"`

2. Install STORM in editable mode (system python):

   `python -m pip install -e .`

### Running Examples

1. Isaac Sim smoketest (spawns Panda, steps physics, prints q/qd):

   `python scripts/isaacsim_smoketest_panda.py --headless`

   If Nucleus assets aren't available, use the bundled URDF path instead:

   `python scripts/isaacsim_smoketest_panda.py --headless --no-use-nucleus-usd`

   For a GUI run with reduced rendering cost:

   `python scripts/isaacsim_smoketest_panda.py --lite`

2. STORM MPPI example (Franka reacher loop):

   `python examples/franka_reacher.py --headless --steps 200`

   If Nucleus assets aren't available:

   `python examples/franka_reacher.py --headless --steps 200 --no-use-nucleus-usd`

   For a GUI run with reduced rendering cost (and to keep the GPU freer for rendering):

   `python examples/franka_reacher.py --lite --no-cuda`

   Note: if you see warnings about missing `weights/robot_self/franka_self_sdf.pt`, STORM will fall back to an
   analytic self-collision check (slower). To generate the NN weights, run:

   `python scripts/train_self_collision.py`
