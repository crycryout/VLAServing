from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import time


REPO_ROOT = pathlib.Path("/root/autodl-tmp/lerobot")
PYTHON = REPO_ROOT / ".venv/bin/python"


@dataclasses.dataclass(frozen=True)
class Config:
    suites: tuple[str, ...] = ("libero_spatial", "libero_goal", "libero_object", "libero_10")
    n_episodes: int = 1
    seed: int = 7
    selector_mode: str = "official_bidir"
    sampling_step: int = 3
    output_root: str = "/root/_runs"


def run_suite(run_root: pathlib.Path, suite: str, cfg: Config) -> dict[str, object]:
    out_dir = run_root / suite
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())

    cmd = [
        str(PYTHON),
        "-m",
        "lerobot.eval.libero_native_eval_pi05",
        "--policy_path",
        "lerobot/pi05_libero_finetuned",
        "--suite",
        suite,
        "--task_ids",
        "[0,1,2,3,4,5,6,7,8,9]",
        "--n_episodes",
        str(cfg.n_episodes),
        "--output_dir",
        str(out_dir),
        "--device",
        "cuda",
        "--dtype",
        "bfloat16",
        "--enable_autohorizon",
        "--selector_mode",
        cfg.selector_mode,
        "--sampling_step",
        str(cfg.sampling_step),
        "--record_autohorizon_events",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return json.loads(summary_path.read_text())


def main(cfg: Config) -> None:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = pathlib.Path(cfg.output_root) / f"lerobot_p50_autohorizon_suites_{cfg.selector_mode}_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for suite in cfg.suites:
        summary = run_suite(run_root, suite, cfg)
        rows.append(
            {
                "suite": suite,
                "pc_success": summary["overall"]["pc_success"],
                "avg_autohorizon": summary["overall"]["avg_autohorizon"],
                "p50_autohorizon": summary["overall"]["p50_autohorizon"],
                "counts": summary.get("autohorizon_artifacts", {}).get("counts", []),
                "suite_dir": str(run_root / suite),
            }
        )
        (run_root / "summary.json").write_text(json.dumps({"run_root": str(run_root), "rows": rows}, indent=2))

    print(json.dumps({"run_root": str(run_root), "rows": rows}, indent=2))


def _parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--selector_mode", type=str, default="official_bidir")
    parser.add_argument("--sampling_step", type=int, default=3)
    parser.add_argument("--output_root", type=str, default="/root/_runs")
    args = parser.parse_args()
    return Config(
        n_episodes=args.n_episodes,
        seed=args.seed,
        selector_mode=args.selector_mode,
        sampling_step=args.sampling_step,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main(_parse_args())
