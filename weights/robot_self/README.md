Contains MLP weights for robot self collision checking.

If the weights file referenced by a task config is missing (e.g. `franka_self_sdf.pt`), STORM will fall back to an
analytic self-collision check (slower).

To generate weights (requires a CUDA-capable PyTorch install):

`python scripts/train_self_collision.py`
