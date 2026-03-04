#!/usr/bin/env python3
import copy
import sys
from pathlib import Path

import torch


def _storm_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    root = _storm_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

def _build_controller(
    *,
    cat_pc_enabled: bool,
    cat_sb_enabled: bool,
    primitive_collision_weight: float | None = None,
    state_bound_weight: float | None = None,
):
    _ensure_imports()
    from storm_kit.mpc.control.mppi import MPPI
    from storm_kit.mpc.rollout.arm_reacher import ArmReacher
    from storm_kit.util_file import get_gym_configs_path, get_mpc_configs_path, join_path, load_yaml

    tensor_args = {"device": torch.device("cpu"), "dtype": torch.float32}

    exp_params = load_yaml(join_path(get_mpc_configs_path(), "franka_reacher.yml"))
    exp_params = copy.deepcopy(exp_params)
    exp_params["robot_params"] = copy.deepcopy(exp_params["model"])

    if primitive_collision_weight is not None:
        exp_params["cost"]["primitive_collision"]["weight"] = float(primitive_collision_weight)
    if state_bound_weight is not None:
        exp_params["cost"]["state_bound"]["weight"] = float(state_bound_weight)

    mppi_cfg = exp_params["mppi"]
    mppi_cfg["horizon"] = 8
    mppi_cfg["num_particles"] = 64
    mppi_cfg["n_iters"] = 1
    mppi_cfg["update_cov"] = False
    mppi_cfg["cov_type"] = "diag_AxA"
    mppi_cfg["sample_params"] = {
        "type": "random",
        "fixed_samples": True,
        "seed": 0,
        "filter_coeffs": None,
    }
    mppi_cfg["cat_primitive_collision"] = {
        "enabled": bool(cat_pc_enabled),
        "p_max": 0.15,
        "tau_c": 0.95,
        "tau_b": 0.95,
        "eps": 1.0e-6,
    }
    mppi_cfg["cat_state_bound"] = {
        "enabled": bool(cat_sb_enabled),
        "p_max": 0.15,
        "tau_c": 0.95,
        "eps": 1.0e-6,
    }

    world_params = load_yaml(join_path(get_gym_configs_path(), "collision_primitives_3d.yml"))
    rollout_fn = ArmReacher(exp_params=exp_params, tensor_args=tensor_args, world_params=world_params)

    n_dofs = rollout_fn.n_dofs
    init_q = torch.as_tensor(exp_params["model"]["init_state"], **tensor_args).unsqueeze(0)
    init_qd = torch.zeros((1, n_dofs), **tensor_args)
    goal_ee_pos, goal_ee_rot = rollout_fn.dynamics_model.robot_model.compute_forward_kinematics(
        init_q,
        init_qd,
        link_name=exp_params["model"]["ee_link_name"],
    )
    rollout_fn.update_params(
        goal_ee_pos=goal_ee_pos.squeeze(0),
        goal_ee_rot=goal_ee_rot.squeeze(0),
    )

    mppi_cfg["d_action"] = rollout_fn.dynamics_model.d_action
    mppi_cfg["action_lows"] = -exp_params["model"]["max_acc"] * torch.ones(mppi_cfg["d_action"], **tensor_args)
    mppi_cfg["action_highs"] = exp_params["model"]["max_acc"] * torch.ones(mppi_cfg["d_action"], **tensor_args)
    mppi_cfg["init_mean"] = torch.zeros((mppi_cfg["horizon"], mppi_cfg["d_action"]), **tensor_args)
    mppi_cfg["rollout_fn"] = rollout_fn
    mppi_cfg["tensor_args"] = tensor_args

    controller = MPPI(**mppi_cfg)

    state = torch.zeros((1, 3 * n_dofs + 1), **tensor_args)
    state[0, :n_dofs] = init_q.squeeze(0)
    state[0, -1] = 0.0

    return controller, state


def _delta_from_violation(violation: torch.Tensor, p_max: float, eps: float) -> torch.Tensor:
    cmax = torch.clamp(torch.max(violation), min=eps)
    delta = p_max * torch.clamp(violation / cmax, min=0.0, max=1.0)
    return torch.clamp(delta, min=0.0, max=p_max)


def _check_survival(survival: torch.Tensor, name: str) -> None:
    if float(torch.min(survival).item()) < -1.0e-7 or float(torch.max(survival).item()) > (1.0 + 1.0e-7):
        raise RuntimeError(f"{name}: survival out of [0,1].")
    if survival.shape[1] > 1:
        max_inc = float(torch.max(survival[:, 1:] - survival[:, :-1]).item())
        if max_inc > 1.0e-6:
            raise RuntimeError(f"{name}: survival is not non-increasing (max increase {max_inc:.3e}).")


def _validate_disabled_path() -> None:
    controller, state = _build_controller(cat_pc_enabled=False, cat_sb_enabled=False)
    traj = controller.generate_rollouts(state)

    if "costs" not in traj:
        raise RuntimeError("disabled path: rollout missing 'costs'.")
    for key in (
        "primitive_collision_costs",
        "primitive_collision_violation",
        "state_bound_costs",
        "state_bound_violation",
    ):
        if key in traj:
            raise RuntimeError(f"disabled path: rollout unexpectedly contains '{key}'.")

    controller._update_distribution(traj)
    controller.optimize(state, shift_steps=0, n_iters=1)
    print("[PASS] cats disabled: rollout + MPPI update succeeded.")


def _validate_state_bound_only_path() -> None:
    controller, state = _build_controller(
        cat_pc_enabled=False,
        cat_sb_enabled=True,
        primitive_collision_weight=1.0,
        state_bound_weight=0.0,
    )
    traj = controller.generate_rollouts(state)

    for key in ("costs", "state_bound_costs", "state_bound_violation"):
        if key not in traj:
            raise RuntimeError(f"state-bound-only path: rollout missing '{key}'.")

    costs = traj["costs"]
    state_bound_costs = traj["state_bound_costs"]
    state_bound_violation = traj["state_bound_violation"]

    if costs.shape != state_bound_costs.shape or costs.shape != state_bound_violation.shape:
        raise RuntimeError(
            "state-bound-only path: shape mismatch "
            f"costs={tuple(costs.shape)}, state_bound_costs={tuple(state_bound_costs.shape)}, "
            f"state_bound_violation={tuple(state_bound_violation.shape)}."
        )

    delta_sb = _delta_from_violation(state_bound_violation, controller._cat_sb_p_max, controller._cat_sb_eps)
    if float(torch.min(delta_sb).item()) < -1.0e-7 or float(torch.max(delta_sb).item()) > (controller._cat_sb_p_max + 1.0e-7):
        raise RuntimeError("state-bound-only path: delta_sb out of [0, p_max].")
    survival = torch.cumprod(1.0 - delta_sb, dim=1)
    _check_survival(survival, "state-bound-only path")

    controller._update_distribution(traj)
    debug = controller._cat_pc_debug
    expected_task_cost_max = float(torch.max(costs - state_bound_costs).item())
    observed_task_cost_max = float(debug.get("cat_task_cost_max", float("nan")))
    if abs(expected_task_cost_max - observed_task_cost_max) > 1.0e-5:
        raise RuntimeError(
            "state-bound-only path: shifted task-cost max mismatch. "
            f"expected={expected_task_cost_max:.6f}, observed={observed_task_cost_max:.6f}."
        )
    controller.optimize(state, shift_steps=0, n_iters=1)
    print("[PASS] cat_state_bound only: rollout breakdown + shifted-CaT MPPI update succeeded.")


def _validate_combined_path() -> None:
    controller, state = _build_controller(
        cat_pc_enabled=True,
        cat_sb_enabled=True,
        primitive_collision_weight=1.0,
        state_bound_weight=1000.0,
    )
    traj = controller.generate_rollouts(state)

    for key in (
        "costs",
        "primitive_collision_costs",
        "primitive_collision_violation",
        "state_bound_costs",
        "state_bound_violation",
    ):
        if key not in traj:
            raise RuntimeError(f"combined path: rollout missing '{key}'.")

    costs = traj["costs"]
    primitive_costs = traj["primitive_collision_costs"]
    primitive_violation = traj["primitive_collision_violation"]
    state_bound_costs = traj["state_bound_costs"]
    state_bound_violation = traj["state_bound_violation"]

    for name, tensor in (
        ("primitive_collision_costs", primitive_costs),
        ("primitive_collision_violation", primitive_violation),
        ("state_bound_costs", state_bound_costs),
        ("state_bound_violation", state_bound_violation),
    ):
        if tensor.shape != costs.shape:
            raise RuntimeError(
                f"combined path: shape mismatch for {name}. "
                f"costs={tuple(costs.shape)}, {name}={tuple(tensor.shape)}."
            )

    delta_pc = _delta_from_violation(primitive_violation, controller._cat_pc_p_max, controller._cat_pc_eps)
    delta_sb = _delta_from_violation(state_bound_violation, controller._cat_sb_p_max, controller._cat_sb_eps)
    delta_expected = torch.maximum(delta_pc, delta_sb)
    survival = torch.cumprod(1.0 - delta_expected, dim=1)
    _check_survival(survival, "combined path")

    controller._update_distribution(traj)
    debug = controller._cat_pc_debug
    delta_max_err = float(debug.get("cat_delta_max_error", float("inf")))
    if delta_max_err > 1.0e-6:
        raise RuntimeError(
            f"combined path: delta combination is not elementwise max (max error {delta_max_err:.3e})."
        )

    observed_delta_max = float(debug.get("cat_delta_max", float("nan")))
    expected_delta_max = float(torch.max(delta_expected).item())
    if abs(observed_delta_max - expected_delta_max) > 1.0e-5:
        raise RuntimeError(
            "combined path: combined delta max mismatch. "
            f"expected={expected_delta_max:.6f}, observed={observed_delta_max:.6f}."
        )

    expected_task_cost_max = float(torch.max(costs - primitive_costs - state_bound_costs).item())
    observed_task_cost_max = float(debug.get("cat_task_cost_max", float("nan")))
    if abs(expected_task_cost_max - observed_task_cost_max) > 1.0e-5:
        raise RuntimeError(
            "combined path: shifted task-cost max mismatch. "
            f"expected={expected_task_cost_max:.6f}, observed={observed_task_cost_max:.6f}."
        )

    controller.optimize(state, shift_steps=0, n_iters=1)
    print("[PASS] cat_primitive_collision + cat_state_bound: combined hazard + shifted-CaT update succeeded.")


def _validate_zero_weight_paths() -> None:
    controller_pc, state_pc = _build_controller(
        cat_pc_enabled=True,
        cat_sb_enabled=False,
        primitive_collision_weight=0.0,
    )
    traj_pc = controller_pc.generate_rollouts(state_pc)
    for key in ("costs", "primitive_collision_costs", "primitive_collision_violation"):
        if key not in traj_pc:
            raise RuntimeError(f"primitive-zero-weight path: rollout missing '{key}'.")
    if traj_pc["costs"].shape != traj_pc["primitive_collision_violation"].shape:
        raise RuntimeError(
            "primitive-zero-weight path: primitive violation shape mismatch "
            f"costs={tuple(traj_pc['costs'].shape)}, "
            f"primitive_collision_violation={tuple(traj_pc['primitive_collision_violation'].shape)}."
        )
    controller_pc._update_distribution(traj_pc)
    controller_pc.optimize(state_pc, shift_steps=0, n_iters=1)

    controller_both, state_both = _build_controller(
        cat_pc_enabled=True,
        cat_sb_enabled=True,
        primitive_collision_weight=0.0,
        state_bound_weight=0.0,
    )
    traj_both = controller_both.generate_rollouts(state_both)
    for key in (
        "costs",
        "primitive_collision_costs",
        "primitive_collision_violation",
        "state_bound_costs",
        "state_bound_violation",
    ):
        if key not in traj_both:
            raise RuntimeError(f"both-zero-weight path: rollout missing '{key}'.")
    controller_both._update_distribution(traj_both)
    controller_both.optimize(state_both, shift_steps=0, n_iters=1)
    print("[PASS] zero additive weights: cat_state_bound and cat_primitive_collision still run.")


def main() -> int:
    try:
        _validate_disabled_path()
        _validate_state_bound_only_path()
        _validate_combined_path()
        _validate_zero_weight_paths()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] shifted-CaT validation failed: {exc}")
        return 1
    print("[PASS] shifted-CaT validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
