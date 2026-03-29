from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class AutoHorizonConfig:
    selector_mode: str = "legacy"
    sampling_step: int = 3
    attn_step_count: int | None = None
    entropy_quantile: float = 0.9
    max_entropy_q: float | None = None
    entropy_threshold: float | None = None
    delta_threshold: float = 0.3
    hold_thr: float | None = None
    run_len: int = 1
    pointer_temperature: float = 0.15
    min_horizon: int = 1
    max_horizon: int | None = None
    prefix_avg_steps: int = 1
    eps: float = 1e-6


def resolve_sampling_step(cfg: AutoHorizonConfig) -> int:
    return max(1, int(cfg.attn_step_count if cfg.attn_step_count is not None else cfg.sampling_step))


def _entropy_quantile(cfg: AutoHorizonConfig) -> float:
    return float(cfg.max_entropy_q if cfg.max_entropy_q is not None else cfg.entropy_quantile)


def _hold_threshold(cfg: AutoHorizonConfig) -> float:
    return float(cfg.hold_thr if cfg.hold_thr is not None else cfg.delta_threshold)


def _safe_row_normalize(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    row_sum = matrix.sum(dim=-1, keepdim=True).clamp_min(eps)
    return matrix / row_sum


def _extract_action_self_attention(attentions: tuple[torch.Tensor, ...], suffix_len: int, eps: float) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    for attn in attentions:
        if attn is None:
            continue
        if attn.ndim == 4:
            blocks.append(attn.mean(dim=(0, 1)))
        elif attn.ndim == 2:
            blocks.append(attn)
    if not blocks:
        raise ValueError("No action self-attention blocks available for AutoHorizon")
    avg = torch.stack(blocks, dim=0).mean(dim=0)
    if avg.shape != (suffix_len, suffix_len):
        raise ValueError(f"Expected action self-attention block of shape {(suffix_len, suffix_len)}, got {tuple(avg.shape)}")
    return _safe_row_normalize(avg.to(dtype=torch.float32), eps)


def _row_entropy(matrix: torch.Tensor, eps: float) -> torch.Tensor:
    if matrix.shape[-1] <= 1:
        return torch.zeros(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    safe = matrix.clamp_min(eps)
    return -(safe * safe.log()).sum(dim=-1) / math.log(matrix.shape[-1])


def _first_true_index(mask: torch.Tensor, default: int) -> int:
    indices = torch.nonzero(mask, as_tuple=False)
    if len(indices) == 0:
        return default
    return int(indices[0].item())


def _make_empty_state(*, horizon_cap: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "selected_rows_mask": torch.zeros(horizon_cap, dtype=torch.bool, device=device),
        "row_entropy": torch.zeros(horizon_cap, dtype=dtype, device=device),
        "expected_trace": torch.zeros(horizon_cap, dtype=dtype, device=device),
        "monotone_expected_trace": torch.zeros(horizon_cap, dtype=dtype, device=device),
        "delta_trace": torch.zeros(horizon_cap, dtype=dtype, device=device),
        "plateau_rows_mask": torch.zeros(horizon_cap, dtype=torch.bool, device=device),
        "stop_row": torch.tensor(max(horizon_cap - 1, 0), dtype=torch.int64, device=device),
        "paper_horizon": torch.tensor(max(horizon_cap, 1), dtype=torch.int64, device=device),
        "entropy_threshold": torch.tensor(0.0, dtype=dtype, device=device),
        "quantile_threshold": torch.tensor(0.0, dtype=dtype, device=device),
    }


def _estimate_prefix_state(matrix: torch.Tensor, cfg: AutoHorizonConfig) -> dict[str, torch.Tensor]:
    matrix = _safe_row_normalize(matrix.to(dtype=torch.float32), cfg.eps)
    horizon_cap = matrix.shape[-1]
    device = matrix.device
    dtype = matrix.dtype
    horizon_cap_t = torch.tensor(horizon_cap, device=device, dtype=torch.int64)

    row_entropy = _row_entropy(matrix, cfg.eps)
    quantile_threshold = torch.quantile(row_entropy.float(), q=min(_entropy_quantile(cfg), 0.999)).to(dtype=dtype)
    if cfg.entropy_threshold is None:
        entropy_threshold = quantile_threshold
    else:
        entropy_threshold = torch.tensor(float(cfg.entropy_threshold), dtype=dtype, device=device)

    selected_rows_mask = row_entropy <= entropy_threshold
    col_idx = torch.arange(horizon_cap, device=device, dtype=dtype)
    expected_trace = (matrix * col_idx[None, :]).sum(dim=-1)
    monotone_expected_trace = torch.cummax(expected_trace, dim=0).values

    prev = torch.cat([monotone_expected_trace.new_tensor([0.0]), monotone_expected_trace[:-1]])
    delta_trace = monotone_expected_trace - prev
    if delta_trace.numel() > 0:
        delta_trace[0] = monotone_expected_trace[0]

    plateau_rows_mask = (delta_trace < _hold_threshold(cfg)) & selected_rows_mask
    run_len = max(1, int(cfg.run_len))
    if run_len == 1:
        stop_row = _first_true_index(plateau_rows_mask, horizon_cap - 1)
    else:
        window = F.conv1d(
            plateau_rows_mask.float()[None, None, :],
            torch.ones(1, 1, run_len, device=device, dtype=torch.float32),
        ).squeeze()
        stop_row = _first_true_index(window >= run_len, horizon_cap - 1)

    paper_horizon = torch.clamp(torch.floor(monotone_expected_trace[min(stop_row, horizon_cap - 1)]).to(torch.int64) + 1, 1, horizon_cap)
    return {
        "selected_rows_mask": selected_rows_mask,
        "row_entropy": row_entropy,
        "expected_trace": expected_trace,
        "monotone_expected_trace": monotone_expected_trace,
        "delta_trace": delta_trace,
        "plateau_rows_mask": plateau_rows_mask,
        "stop_row": torch.tensor(stop_row, dtype=torch.int64, device=device),
        "paper_horizon": torch.where(horizon_cap_t > 0, paper_horizon, torch.ones((), dtype=torch.int64, device=device)),
        "entropy_threshold": entropy_threshold,
        "quantile_threshold": quantile_threshold,
    }


def _map_backward_state(reversed_state: dict[str, torch.Tensor], horizon_cap: int) -> dict[str, torch.Tensor]:
    device = reversed_state["expected_trace"].device
    dtype = reversed_state["expected_trace"].dtype
    horizon_cap_minus_one = torch.tensor(horizon_cap - 1, dtype=dtype, device=device)

    expected_trace = torch.flip(horizon_cap_minus_one - reversed_state["expected_trace"], dims=[0])
    monotone_expected_trace = torch.flip(horizon_cap_minus_one - reversed_state["monotone_expected_trace"], dims=[0])
    stop_row = max(horizon_cap - 1 - int(reversed_state["stop_row"].item()), 0)
    paper_horizon = torch.clamp(torch.floor(monotone_expected_trace[stop_row]).to(torch.int64) + 1, 1, horizon_cap)
    return {
        "selected_rows_mask": torch.flip(reversed_state["selected_rows_mask"], dims=[0]),
        "row_entropy": torch.flip(reversed_state["row_entropy"], dims=[0]),
        "expected_trace": expected_trace,
        "monotone_expected_trace": monotone_expected_trace,
        "delta_trace": torch.flip(reversed_state["delta_trace"], dims=[0]),
        "plateau_rows_mask": torch.flip(reversed_state["plateau_rows_mask"], dims=[0]),
        "stop_row": torch.tensor(stop_row, dtype=torch.int64, device=device),
        "paper_horizon": paper_horizon,
        "entropy_threshold": reversed_state["entropy_threshold"],
        "quantile_threshold": reversed_state["quantile_threshold"],
    }


def _fallback_autohorizon_state(*, suffix_len: int, cfg: AutoHorizonConfig) -> dict[str, torch.Tensor]:
    device = torch.device("cpu")
    dtype = torch.float32
    horizon = min(int(cfg.max_horizon or suffix_len), suffix_len)
    horizon = max(int(cfg.min_horizon), horizon)
    state = _make_empty_state(horizon_cap=suffix_len, device=device, dtype=dtype)
    state.update(
        {
            "execution_horizon": torch.tensor(horizon, dtype=torch.int64, device=device),
            "paper_execution_horizon": torch.tensor(horizon, dtype=torch.int64, device=device),
            "prediction_horizon": torch.tensor(suffix_len, dtype=torch.int64, device=device),
            "full_range_coverage": torch.tensor(False, dtype=torch.bool, device=device),
            "forward_horizon": torch.tensor(horizon, dtype=torch.int64, device=device),
            "backward_horizon": torch.tensor(0, dtype=torch.int64, device=device),
            "forward_entropy_threshold": state["entropy_threshold"],
            "backward_entropy_threshold": state["entropy_threshold"].clone(),
            "forward_quantile_threshold": state["quantile_threshold"],
            "backward_quantile_threshold": state["quantile_threshold"].clone(),
            "forward_selected_rows_mask": state["selected_rows_mask"],
            "backward_selected_rows_mask": state["selected_rows_mask"].clone(),
            "forward_row_entropy": state["row_entropy"],
            "backward_row_entropy": state["row_entropy"].clone(),
            "forward_expected_trace": state["expected_trace"],
            "backward_expected_trace": state["expected_trace"].clone(),
            "forward_monotone_expected_trace": state["monotone_expected_trace"],
            "backward_monotone_expected_trace": state["monotone_expected_trace"].clone(),
            "forward_delta_trace": state["delta_trace"],
            "backward_delta_trace": state["delta_trace"].clone(),
            "forward_plateau_rows_mask": state["plateau_rows_mask"],
            "backward_plateau_rows_mask": state["plateau_rows_mask"].clone(),
            "forward_stop_row": state["stop_row"],
            "backward_stop_row": state["stop_row"].clone(),
            "join_row": torch.tensor(-1, dtype=torch.int64, device=device),
            "gap": torch.zeros(suffix_len, dtype=dtype, device=device),
        }
    )
    return state


def _official_soft_pointer_prefix(
    A: torch.Tensor,
    *,
    hold_thr: float,
    run_len: int,
    max_entropy_q: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    assert A.ndim == 2 and A.size(0) == A.size(1)
    dev, T = A.device, A.size(0)
    eps = 1e-12

    A = A / (A.sum(-1, keepdim=True) + eps)
    idx = torch.arange(T, device=dev, dtype=A.dtype)
    mu = (A * idx).sum(-1)
    mu = torch.maximum(mu, torch.cummax(mu, dim=0).values)

    ent = -(A.clamp_min(eps) * (A.clamp_min(eps)).log()).sum(-1) / math.log(T)
    thr_ent = torch.quantile(ent.float(), q=min(max_entropy_q, 0.999)).to(dtype=A.dtype)
    reliable = ent <= thr_ent

    prev = torch.cat([mu.new_tensor([0.0]), mu[:-1]])
    dmu = mu - prev
    dmu[0] = mu[0]

    is_hold = (dmu < hold_thr) & reliable
    run_len = max(1, int(run_len))
    if run_len == 1:
        stop_row = _first_true_index(is_hold, T - 1)
    else:
        window = F.conv1d(is_hold.float()[None, None, :], torch.ones(1, 1, run_len, device=dev)).squeeze()
        if window.numel() and (window >= run_len - 1e-6).any():
            stop_row = int(torch.nonzero(window >= run_len, as_tuple=False)[0].item())
        else:
            stop_row = T - 1

    horizon = torch.clamp(torch.floor(mu[min(stop_row, T - 1)]).to(torch.int64) + 1, 1, T)
    return mu, dmu, reliable, ent, thr_ent, stop_row, horizon


def _estimate_execution_horizon_state_official(
    matrix: torch.Tensor,
    cfg: AutoHorizonConfig,
) -> dict[str, torch.Tensor]:
    matrix = _safe_row_normalize(matrix.to(dtype=torch.float32), cfg.eps)
    device = matrix.device
    T = matrix.shape[-1]
    hold_thr = float(_hold_threshold(cfg))
    max_entropy_q = float(_entropy_quantile(cfg))
    run_len = max(1, int(cfg.run_len))

    mu_f, dmu_f, rel_f, ent_f, thr_f, stop_f, N_f = _official_soft_pointer_prefix(
        matrix,
        hold_thr=hold_thr,
        run_len=run_len,
        max_entropy_q=max_entropy_q,
    )

    rev_matrix = torch.flip(matrix, dims=(0, 1))
    mu_b_rev, dmu_b_rev, rel_b_rev, ent_b_rev, thr_b_rev, stop_b_rev, _ = _official_soft_pointer_prefix(
        rev_matrix,
        hold_thr=hold_thr,
        run_len=run_len,
        max_entropy_q=max_entropy_q,
    )

    idx_dtype = mu_f.dtype
    Tm1 = torch.tensor(T - 1, dtype=idx_dtype, device=device)
    mu_b = torch.flip(Tm1 - mu_b_rev, dims=(0,))
    ent_b = torch.flip(ent_b_rev, dims=(0,))
    rel_b = torch.flip(rel_b_rev, dims=(0,))
    dmu_b = torch.flip(dmu_b_rev, dims=(0,))
    stop_b = max(T - 1 - int(stop_b_rev), 0)
    N_b = torch.clamp(torch.floor(mu_b[stop_b]).to(torch.int64) + 1, 1, T)

    gap = mu_b - mu_f
    meet_mask = gap <= 1.0
    join_row_value = _first_true_index(meet_mask, -1)
    join_row = torch.tensor(join_row_value, dtype=torch.int64, device=device)

    prediction_horizon = torch.tensor(T, dtype=torch.int64, device=device)
    full_range_coverage = (N_f + N_b >= prediction_horizon) & (join_row >= 0)
    paper_horizon = torch.where(full_range_coverage, prediction_horizon, N_f)
    min_horizon = torch.tensor(int(cfg.min_horizon), dtype=torch.int64, device=device)
    max_horizon = torch.tensor(int(cfg.max_horizon or T), dtype=torch.int64, device=device)
    execution_horizon = torch.clamp(paper_horizon, min=min_horizon, max=max_horizon)

    plateau_f = (dmu_f < hold_thr) & rel_f
    plateau_b = (dmu_b < hold_thr) & rel_b

    return {
        "execution_horizon": execution_horizon,
        "paper_execution_horizon": paper_horizon,
        "prediction_horizon": prediction_horizon,
        "full_range_coverage": full_range_coverage,
        "forward_horizon": N_f,
        "backward_horizon": N_b,
        "forward_entropy_threshold": thr_f,
        "backward_entropy_threshold": thr_b_rev.to(device=device, dtype=matrix.dtype),
        "forward_quantile_threshold": thr_f,
        "backward_quantile_threshold": thr_b_rev.to(device=device, dtype=matrix.dtype),
        "forward_selected_rows_mask": rel_f,
        "backward_selected_rows_mask": rel_b,
        "forward_row_entropy": ent_f,
        "backward_row_entropy": ent_b,
        "forward_expected_trace": mu_f,
        "backward_expected_trace": mu_b,
        "forward_monotone_expected_trace": mu_f,
        "backward_monotone_expected_trace": mu_b,
        "forward_delta_trace": dmu_f,
        "backward_delta_trace": dmu_b,
        "forward_plateau_rows_mask": plateau_f,
        "backward_plateau_rows_mask": plateau_b,
        "forward_stop_row": torch.tensor(stop_f, dtype=torch.int64, device=device),
        "backward_stop_row": torch.tensor(stop_b, dtype=torch.int64, device=device),
        "join_row": join_row,
        "gap": gap,
    }


def estimate_execution_horizon_state(
    attentions: tuple[torch.Tensor, ...],
    suffix_len: int,
    cfg: AutoHorizonConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = cfg or AutoHorizonConfig(max_horizon=suffix_len)
    matrix = _extract_action_self_attention(attentions, suffix_len=suffix_len, eps=cfg.eps)
    if cfg.selector_mode == "official_bidir":
        return _estimate_execution_horizon_state_official(matrix, cfg)
    device = matrix.device

    forward_state = _estimate_prefix_state(matrix, cfg)
    backward_state = _map_backward_state(_estimate_prefix_state(torch.flip(matrix, dims=[0, 1]), cfg), matrix.shape[-1])

    prediction_horizon = torch.tensor(matrix.shape[-1], dtype=torch.int64, device=device)
    gap = backward_state["monotone_expected_trace"] - forward_state["monotone_expected_trace"]
    meet_mask = gap <= 1.0
    join_row_value = _first_true_index(meet_mask, -1)
    join_row = torch.tensor(join_row_value, dtype=torch.int64, device=device)
    full_range_coverage = (forward_state["paper_horizon"] + backward_state["paper_horizon"] >= prediction_horizon) & (join_row >= 0)
    paper_horizon = torch.where(full_range_coverage, prediction_horizon, forward_state["paper_horizon"])
    min_horizon = torch.tensor(int(cfg.min_horizon), dtype=torch.int64, device=device)
    max_horizon = torch.tensor(int(cfg.max_horizon or matrix.shape[-1]), dtype=torch.int64, device=device)
    execution_horizon = torch.clamp(paper_horizon, min=min_horizon, max=max_horizon)

    return {
        "execution_horizon": execution_horizon,
        "paper_execution_horizon": paper_horizon,
        "prediction_horizon": prediction_horizon,
        "full_range_coverage": full_range_coverage,
        "forward_horizon": forward_state["paper_horizon"],
        "backward_horizon": backward_state["paper_horizon"],
        "forward_entropy_threshold": forward_state["entropy_threshold"],
        "backward_entropy_threshold": backward_state["entropy_threshold"],
        "forward_quantile_threshold": forward_state["quantile_threshold"],
        "backward_quantile_threshold": backward_state["quantile_threshold"],
        "forward_selected_rows_mask": forward_state["selected_rows_mask"],
        "backward_selected_rows_mask": backward_state["selected_rows_mask"],
        "forward_row_entropy": forward_state["row_entropy"],
        "backward_row_entropy": backward_state["row_entropy"],
        "forward_expected_trace": forward_state["expected_trace"],
        "backward_expected_trace": backward_state["expected_trace"],
        "forward_monotone_expected_trace": forward_state["monotone_expected_trace"],
        "backward_monotone_expected_trace": backward_state["monotone_expected_trace"],
        "forward_delta_trace": forward_state["delta_trace"],
        "backward_delta_trace": backward_state["delta_trace"],
        "forward_plateau_rows_mask": forward_state["plateau_rows_mask"],
        "backward_plateau_rows_mask": backward_state["plateau_rows_mask"],
        "forward_stop_row": forward_state["stop_row"],
        "backward_stop_row": backward_state["stop_row"],
        "join_row": join_row,
        "gap": gap,
    }


def _tensor_scalar_to_int(value: torch.Tensor) -> int:
    return int(value.detach().cpu().item())


def _tensor_scalar_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _materialize_direction(prefix: str, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    selected_rows_mask = state[f"{prefix}_selected_rows_mask"].detach().cpu()
    plateau_rows_mask = state[f"{prefix}_plateau_rows_mask"].detach().cpu()
    return {
        "paper_horizon": _tensor_scalar_to_int(state[f"{prefix}_horizon"]),
        "selected_row_count": int(selected_rows_mask.sum().item()),
        "selected_rows": torch.nonzero(selected_rows_mask, as_tuple=False).flatten().tolist(),
        "row_entropy_mean": float(state[f"{prefix}_row_entropy"].detach().cpu().mean().item()),
        "entropy_threshold": _tensor_scalar_to_float(state[f"{prefix}_entropy_threshold"]),
        "quantile_threshold": _tensor_scalar_to_float(state[f"{prefix}_quantile_threshold"]),
        "expected_trace": [float(x) for x in state[f"{prefix}_expected_trace"].detach().cpu().tolist()],
        "monotone_expected_trace": [
            float(x) for x in state[f"{prefix}_monotone_expected_trace"].detach().cpu().tolist()
        ],
        "delta_trace": [float(x) for x in state[f"{prefix}_delta_trace"].detach().cpu().tolist()],
        "plateau_rows": torch.nonzero(plateau_rows_mask, as_tuple=False).flatten().tolist(),
        "stop_row": _tensor_scalar_to_int(state[f"{prefix}_stop_row"]),
    }


def materialize_autohorizon_meta(state: dict[str, torch.Tensor], cfg: AutoHorizonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AutoHorizonConfig()
    if not state:
        fallback = max(int(cfg.min_horizon), 1)
        return {"execution_horizon": fallback, "per_step_horizons": [], "avg_selected_row_count": 0.0}

    forward_stats = _materialize_direction("forward", state)
    backward_stats = _materialize_direction("backward", state)
    join_row = _tensor_scalar_to_int(state["join_row"])
    execution_horizon = _tensor_scalar_to_int(state["execution_horizon"])
    return {
        "execution_horizon": execution_horizon,
        "paper_execution_horizon": _tensor_scalar_to_int(state["paper_execution_horizon"]),
        "prediction_horizon": _tensor_scalar_to_int(state["prediction_horizon"]),
        "full_range_coverage": bool(state["full_range_coverage"].detach().cpu().item()),
        "forward_horizon": forward_stats["paper_horizon"],
        "backward_horizon": backward_stats["paper_horizon"],
        "selected_row_count": forward_stats["selected_row_count"],
        "selected_rows": forward_stats["selected_rows"],
        "row_entropy_mean": forward_stats["row_entropy_mean"],
        "entropy_threshold": forward_stats["entropy_threshold"],
        "quantile_threshold": forward_stats["quantile_threshold"],
        "expected_trace": forward_stats["expected_trace"],
        "monotone_expected_trace": forward_stats["monotone_expected_trace"],
        "delta_trace": forward_stats["delta_trace"],
        "plateau_rows": forward_stats["plateau_rows"],
        "forward_stop_row": forward_stats["stop_row"],
        "backward_stop_row": backward_stats["stop_row"],
        "join_row": None if join_row < 0 else join_row,
        "gap": [float(x) for x in state["gap"].detach().cpu().tolist()],
        "forward_stats": forward_stats,
        "backward_stats": backward_stats,
        "per_step_horizons": [execution_horizon],
        "avg_selected_row_count": float(forward_stats["selected_row_count"]),
        "entropy_threshold_mean": float(forward_stats["entropy_threshold"]),
        "step_stats": [
            {
                "execution_horizon": execution_horizon,
                "selected_row_count": forward_stats["selected_row_count"],
                "row_entropy_mean": forward_stats["row_entropy_mean"],
                "entropy_threshold": forward_stats["entropy_threshold"],
            }
        ],
    }


def estimate_execution_horizon(
    attentions: tuple[torch.Tensor, ...],
    suffix_len: int,
    cfg: AutoHorizonConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or AutoHorizonConfig(max_horizon=suffix_len)
    state = estimate_execution_horizon_state(attentions=attentions, suffix_len=suffix_len, cfg=cfg)
    return materialize_autohorizon_meta(state, cfg=cfg)


def aggregate_horizons(step_stats: list[dict[str, Any]], cfg: AutoHorizonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AutoHorizonConfig()
    if not step_stats:
        fallback = max(int(cfg.min_horizon), 1)
        return {"execution_horizon": fallback, "per_step_horizons": [], "avg_selected_row_count": 0.0}

    prefix = step_stats[: max(1, int(cfg.prefix_avg_steps))]
    horizons = [int(item["execution_horizon"]) for item in prefix]
    horizon = int(round(sum(horizons) / max(1, len(horizons))))
    horizon = max(int(cfg.min_horizon), min(int(cfg.max_horizon or horizon), horizon))
    return {
        "execution_horizon": int(horizon),
        "per_step_horizons": [int(item["execution_horizon"]) for item in step_stats],
        "avg_selected_row_count": float(
            sum(float(item.get("selected_row_count", 0)) for item in step_stats) / max(1, len(step_stats))
        ),
        "row_entropy_mean": float(
            sum(float(item.get("row_entropy_mean", 0.0)) for item in step_stats) / max(1, len(step_stats))
        ),
        "entropy_threshold_mean": float(
            sum(float(item.get("entropy_threshold", 0.0)) for item in step_stats) / max(1, len(step_stats))
        ),
        "step_stats": step_stats,
    }


def fallback_autohorizon_meta(*, suffix_len: int, cfg: AutoHorizonConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AutoHorizonConfig(max_horizon=suffix_len)
    return materialize_autohorizon_meta(_fallback_autohorizon_state(suffix_len=suffix_len, cfg=cfg), cfg=cfg)


def fallback_autohorizon_state(*, suffix_len: int, cfg: AutoHorizonConfig | None = None) -> dict[str, torch.Tensor]:
    cfg = cfg or AutoHorizonConfig(max_horizon=suffix_len)
    return _fallback_autohorizon_state(suffix_len=suffix_len, cfg=cfg)
