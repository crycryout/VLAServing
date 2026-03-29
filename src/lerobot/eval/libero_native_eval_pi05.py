#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.libero import LiberoEnv as NativeLiberoEnv
from lerobot.envs.libero import _get_suite
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.pi05.configuration_pi05 import PI05Config  # noqa: F401
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyAction
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import get_safe_torch_device, init_logging

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


def _build_policy_obs(
    observation: dict[str, Any],
    *,
    task_text: str,
    env_preprocessor,
    preprocessor,
) -> dict[str, Any]:
    obs = preprocess_observation(observation)
    if "observation.robot_state" in obs:
        obs["observation.robot_state"] = _batchify_nested_tensors(obs["observation.robot_state"])
    obs["task"] = [task_text]
    obs = env_preprocessor(obs)
    obs = preprocessor(obs)
    return obs


def _batchify_nested_tensors(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _batchify_nested_tensors(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor) and value.ndim >= 1:
        return value.unsqueeze(0)
    return value


def _tensor_to_action_chunk(action_chunk: PolicyAction, postprocessor) -> np.ndarray:
    chunk = postprocessor(action_chunk)
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(0)
    chunk_np = chunk.detach().cpu().numpy()
    if chunk_np.ndim != 3 or chunk_np.shape[0] != 1:
        raise ValueError(f"Unexpected action chunk shape: {chunk_np.shape}")
    return np.asarray(chunk_np[0], dtype=np.float32)


def _make_env(suite_name: str, task_id: int, episode_index: int) -> NativeLiberoEnv:
    suite = _get_suite(suite_name)
    return NativeLiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        episode_index=episode_index,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=360,
        observation_height=360,
        init_states=True,
        control_mode="relative",
    )


def _run_episode(
    *,
    env: NativeLiberoEnv,
    policy,
    env_preprocessor,
    preprocessor,
    postprocessor,
    device: torch.device,
    use_amp: bool,
    fixed_budget: int | None,
    autohorizon_cfg: dict[str, Any] | None,
    max_steps: int,
    log_inference_steps: bool,
    log_env_steps: bool,
    record_autohorizon_events: bool,
) -> dict[str, Any]:
    policy.reset()
    logging.info("Episode reset start | task=%s | episode_index=%s", env.task_id, env.episode_index)
    observation, _ = env.reset()
    logging.info("Episode reset done | task=%s | episode_index=%s", env.task_id, env.episode_index)

    action_queue: deque[np.ndarray] = deque()
    infer_ms: list[float] = []
    autoh_values: list[int] = []
    sum_reward = 0.0
    max_reward = 0.0
    success = False
    consumed_actions_since_request = 0
    step_count = 0
    episode_t0 = time.perf_counter()
    autohorizon_events: list[dict[str, Any]] = []

    while step_count < max_steps:
        active_budget = fixed_budget
        if fixed_budget is None:
            active_budget = 1 if not autoh_values else autoh_values[-1]

        should_request = len(action_queue) == 0 or consumed_actions_since_request >= active_budget
        if should_request:
            if log_inference_steps:
                logging.info("Inference request start | task=%s | step=%s", env.task_id, step_count)
            obs_t = _build_policy_obs(
                observation,
                task_text=env.task_description,
                env_preprocessor=env_preprocessor,
                preprocessor=preprocessor,
            )
            if log_inference_steps:
                logging.info("Inference preprocessing done | task=%s | step=%s", env.task_id, step_count)
            ctx = torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext()
            t0 = time.perf_counter()
            with torch.inference_mode(), ctx:
                if fixed_budget is None:
                    action_chunk_t, meta = policy.predict_action_chunk_with_autohorizon(
                        obs_t,
                        autohorizon_cfg=autohorizon_cfg,
                    )
                    active_budget = max(1, int(meta["execution_horizon"]))
                    autoh_values.append(active_budget)
                else:
                    action_chunk_t = policy.predict_action_chunk(obs_t)
            infer_ms.append((time.perf_counter() - t0) * 1000.0)
            if log_inference_steps:
                logging.info(
                    "Inference request done | task=%s | step=%s | infer_ms=%.2f",
                    env.task_id,
                    step_count,
                    infer_ms[-1],
                )
            if fixed_budget is None and record_autohorizon_events:
                autohorizon_events.append(
                    {
                        "task_id": int(env.task_id),
                        "episode_index": int(env.episode_index),
                        "decision_index": len(autohorizon_events),
                        "env_step": int(step_count),
                        "episode_elapsed_sec": float(time.perf_counter() - episode_t0),
                        "infer_ms": float(infer_ms[-1]),
                        "execution_horizon": int(active_budget),
                    }
                )
            action_chunk = _tensor_to_action_chunk(action_chunk_t, postprocessor)
            action_queue.clear()
            action_queue.extend(action_chunk)
            consumed_actions_since_request = 0

        if not action_queue:
            raise RuntimeError("Action queue is empty after inference.")

        action = action_queue.popleft()
        observation, reward, terminated, truncated, info = env.step(action)
        if log_env_steps:
            logging.info(
                "Env step done | task=%s | step=%s | reward=%.4f | terminated=%s | truncated=%s",
                env.task_id,
                step_count,
                float(reward),
                terminated,
                truncated,
            )

        reward_val = float(reward)
        sum_reward += reward_val
        max_reward = max(max_reward, reward_val)
        consumed_actions_since_request += 1
        step_count += 1

        if terminated or truncated:
            success = bool(info.get("is_success", False))
            break

    return {
        "sum_reward": sum_reward,
        "max_reward": max_reward,
        "success": success,
        "steps": step_count,
        "avg_infer_ms": float(np.mean(infer_ms)) if infer_ms else 0.0,
        "avg_autohorizon": float(np.mean(autoh_values)) if autoh_values else 0.0,
        "p50_autohorizon": float(np.percentile(autoh_values, 50)) if autoh_values else 0.0,
        "autohorizon_events": autohorizon_events,
    }


def _summarize(task_rows: list[dict[str, Any]], suite_name: str) -> dict[str, Any]:
    all_eps = [ep for row in task_rows for ep in row["episodes"]]
    overall = {
        "pc_success": float(np.mean([ep["success"] for ep in all_eps]) * 100.0) if all_eps else 0.0,
        "n_episodes": len(all_eps),
        "avg_sum_reward": float(np.mean([ep["sum_reward"] for ep in all_eps])) if all_eps else 0.0,
        "avg_max_reward": float(np.mean([ep["max_reward"] for ep in all_eps])) if all_eps else 0.0,
        "avg_server_infer_ms": float(np.mean([ep["avg_infer_ms"] for ep in all_eps])) if all_eps else 0.0,
        "avg_autohorizon": float(np.mean([ep["avg_autohorizon"] for ep in all_eps if ep["avg_autohorizon"] > 0]))
        if any(ep["avg_autohorizon"] > 0 for ep in all_eps)
        else 0.0,
        "p50_autohorizon": float(
            np.percentile([ep["avg_autohorizon"] for ep in all_eps if ep["avg_autohorizon"] > 0], 50)
        )
        if any(ep["avg_autohorizon"] > 0 for ep in all_eps)
        else 0.0,
    }
    per_suite = {
        suite_name: {
            "pc_success": overall["pc_success"],
            "n_episodes": overall["n_episodes"],
            "avg_sum_reward": overall["avg_sum_reward"],
            "avg_max_reward": overall["avg_max_reward"],
            "avg_server_infer_ms": overall["avg_server_infer_ms"],
            "avg_autohorizon": overall["avg_autohorizon"],
            "p50_autohorizon": overall["p50_autohorizon"],
        }
    }
    return {"per_task": task_rows, "per_suite": per_suite, "overall": overall}


def _write_autohorizon_artifacts(task_rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any] | None:
    events = [
        {
            **event,
            "task_group": row["task_group"],
        }
        for row in task_rows
        for episode in row["episodes"]
        for event in episode.get("autohorizon_events", [])
    ]
    if not events:
        return None

    events.sort(key=lambda event: (event["task_id"], event["episode_index"], event["decision_index"]))
    cumulative_step = 0
    for global_index, event in enumerate(events):
        event["global_decision_index"] = int(global_index)
        event["suite_cumulative_env_step"] = int(cumulative_step + event["env_step"])
        if global_index + 1 < len(events):
            same_episode = (
                events[global_index + 1]["task_id"] == event["task_id"]
                and events[global_index + 1]["episode_index"] == event["episode_index"]
            )
            if not same_episode:
                matching_episode = next(
                    episode
                    for row in task_rows
                    if row["task_id"] == event["task_id"]
                    for episode in row["episodes"]
                    if episode["autohorizon_events"]
                    and episode["autohorizon_events"][0]["task_id"] == event["task_id"]
                    and episode["autohorizon_events"][0]["episode_index"] == event["episode_index"]
                )
                cumulative_step += int(matching_episode["steps"])
        else:
            matching_episode = next(
                episode
                for row in task_rows
                if row["task_id"] == event["task_id"]
                for episode in row["episodes"]
                if episode["autohorizon_events"]
                and episode["autohorizon_events"][0]["task_id"] == event["task_id"]
                and episode["autohorizon_events"][0]["episode_index"] == event["episode_index"]
            )
            cumulative_step += int(matching_episode["steps"])

    with (output_dir / "autohorizon_events.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_group",
                "task_id",
                "episode_index",
                "decision_index",
                "global_decision_index",
                "env_step",
                "suite_cumulative_env_step",
                "episode_elapsed_sec",
                "infer_ms",
                "execution_horizon",
            ],
        )
        writer.writeheader()
        writer.writerows(events)

    horizon_values = [int(event["execution_horizon"]) for event in events]
    unique_horizons, counts = np.unique(np.asarray(horizon_values, dtype=np.int64), return_counts=True)
    total = int(sum(counts))
    count_rows = [
        {
            "execution_horizon": int(horizon),
            "count": int(count),
            "frequency": float(count / total),
        }
        for horizon, count in zip(unique_horizons.tolist(), counts.tolist(), strict=True)
    ]
    with (output_dir / "autohorizon_counts.json").open("w") as f:
        json.dump(
            {
                "n_decisions": total,
                "counts": count_rows,
            },
            f,
            indent=2,
        )
    with (output_dir / "autohorizon_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["execution_horizon", "count", "frequency"])
        writer.writeheader()
        writer.writerows(count_rows)

    if plt is not None:
        x = [event["suite_cumulative_env_step"] for event in events]
        y = [event["execution_horizon"] for event in events]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(x, y, color="#1f77b4", linewidth=1.0, alpha=0.8)
        ax.scatter(x, y, color="#d62728", s=14)
        ax.set_xlabel("Suite Cumulative Env Step")
        ax.set_ylabel("AutoHorizon Execution Horizon")
        ax.set_title("LIBERO AutoHorizon Decisions Over Time")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "autohorizon_timeline.png", dpi=160)
        plt.close(fig)

    return {
        "n_decisions": total,
        "counts": count_rows,
        "events_path": str(output_dir / "autohorizon_events.csv"),
        "counts_path": str(output_dir / "autohorizon_counts.json"),
        "timeline_path": str(output_dir / "autohorizon_timeline.png"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Native non-realtime LIBERO eval for LeRobot PI05.")
    parser.add_argument("--policy_path", type=str, default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--suite", type=str, required=True)
    parser.add_argument("--task_ids", type=str, default="[0,1,2,3,4,5,6,7,8,9]")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=520)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", type=str, default=None)
    parser.add_argument("--compile_disable_cudagraphs", action="store_true")
    parser.add_argument("--fixed_budget", type=int, default=0)
    parser.add_argument("--enable_autohorizon", action="store_true")
    parser.add_argument("--entropy_quantile", type=float, default=0.9)
    parser.add_argument("--delta_threshold", type=float, default=0.3)
    parser.add_argument("--sampling_step", type=int, default=3)
    parser.add_argument("--selector_mode", type=str, default="legacy", choices=["legacy", "official_bidir"])
    parser.add_argument("--log_inference_steps", action="store_true")
    parser.add_argument("--log_env_steps", action="store_true")
    parser.add_argument("--record_autohorizon_events", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logging(log_file=output_dir / "eval.log")

    task_ids = json.loads(args.task_ids)
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty JSON list.")
    if args.enable_autohorizon and args.fixed_budget > 0:
        raise ValueError("fixed_budget and enable_autohorizon are mutually exclusive.")
    if not args.enable_autohorizon and args.fixed_budget <= 0:
        raise ValueError("fixed_budget must be > 0 when autohorizon is disabled.")

    device = get_safe_torch_device(args.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = args.policy_path
    policy_cfg.device = str(device)
    policy_cfg.dtype = args.dtype
    policy_cfg.n_action_steps = 50
    effective_compile_mode = args.compile_mode
    if args.compile_model and effective_compile_mode is None:
        effective_compile_mode = "reduce-overhead"
    if args.compile_model and args.compile_disable_cudagraphs:
        if effective_compile_mode in {"default", "reduce-overhead", "max-autotune"}:
            effective_compile_mode = f"{effective_compile_mode}-no-cudagraphs"
        os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"
    if hasattr(policy_cfg, "compile_model"):
        policy_cfg.compile_model = bool(args.compile_model)
    if hasattr(policy_cfg, "compile_mode") and effective_compile_mode:
        policy_cfg.compile_mode = effective_compile_mode
    if hasattr(policy_cfg, "gradient_checkpointing"):
        policy_cfg.gradient_checkpointing = False
    logging.info(
        "Policy load config | compile_model=%s | compile_mode=%s | disable_cudagraphs=%s",
        bool(args.compile_model),
        getattr(policy_cfg, "compile_mode", None),
        bool(args.compile_disable_cudagraphs),
    )

    env_cfg = LiberoEnvConfig(task=args.suite)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_cfg.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    autohorizon_cfg = None
    if args.enable_autohorizon:
        autohorizon_cfg = {
            "max_horizon": 50,
            "entropy_quantile": args.entropy_quantile,
            "delta_threshold": args.delta_threshold,
            "sampling_step": args.sampling_step,
            "min_horizon": 1,
            "selector_mode": args.selector_mode,
        }

    task_rows: list[dict[str, Any]] = []
    use_amp = args.dtype == "bfloat16"

    for task_id in task_ids:
        episodes: list[dict[str, Any]] = []
        for episode_index in range(args.n_episodes):
            env = _make_env(args.suite, task_id, episode_index)
            try:
                ep = _run_episode(
                    env=env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    device=device,
                    use_amp=use_amp,
                    fixed_budget=None if args.enable_autohorizon else args.fixed_budget,
                    autohorizon_cfg=autohorizon_cfg,
                    max_steps=args.max_steps,
                    log_inference_steps=bool(args.log_inference_steps),
                    log_env_steps=bool(args.log_env_steps),
                    record_autohorizon_events=bool(args.record_autohorizon_events),
                )
            finally:
                env.close()
            episodes.append(ep)
        task_rows.append(
            {
                "task_group": args.suite,
                "task_id": task_id,
                "metrics": {
                    "pc_success": float(np.mean([ep["success"] for ep in episodes]) * 100.0),
                    "n_episodes": len(episodes),
                    "avg_sum_reward": float(np.mean([ep["sum_reward"] for ep in episodes])),
                    "avg_max_reward": float(np.mean([ep["max_reward"] for ep in episodes])),
                    "avg_server_infer_ms": float(np.mean([ep["avg_infer_ms"] for ep in episodes])),
                    "avg_autohorizon": float(np.mean([ep["avg_autohorizon"] for ep in episodes if ep["avg_autohorizon"] > 0]))
                    if any(ep["avg_autohorizon"] > 0 for ep in episodes)
                    else 0.0,
                    "p50_autohorizon": float(
                        np.percentile([ep["avg_autohorizon"] for ep in episodes if ep["avg_autohorizon"] > 0], 50)
                    )
                    if any(ep["avg_autohorizon"] > 0 for ep in episodes)
                    else 0.0,
                },
                "episodes": episodes,
            }
        )
        logging.info(
            "Task %s/%s success=%.1f%%",
            args.suite,
            task_id,
            task_rows[-1]["metrics"]["pc_success"],
        )

    summary = _summarize(task_rows, args.suite)
    if args.record_autohorizon_events:
        autohorizon_artifacts = _write_autohorizon_artifacts(task_rows, output_dir)
        if autohorizon_artifacts is not None:
            summary["autohorizon_artifacts"] = autohorizon_artifacts
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Summary written to %s", output_dir / "summary.json")
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
