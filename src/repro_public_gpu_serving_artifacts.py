#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/VLAServing")
RESULTS = ROOT / "results"
OUT = RESULTS / "public_gpu_serving_artifacts_20260411.json"

GLET_ROOT = Path("/root/autodl-tmp/glet")
GLET_BIN = GLET_ROOT / "bin" / "standalone_scheduler"
GLET_MEM_CONFIG = GLET_ROOT / "resource" / "mem-config.json"

PI05_AUTOH = RESULTS / "pi05_autohorizon_simulator_fit_20260329.json"
GR00T_AUTOH = RESULTS / "groot_n15_official_horizon_simulator_fit_20260328.json"
PI05_BATCH = RESULTS / "lerobot_p50_step1_full_e2e_batch_sweep_compile_dynamic_20260327_summary.json"
GR00T_BATCH = RESULTS / "groot_n1d6_same_model_batch_curve_step1_compile_libero.json"


FIXED_MODEL_IDS = {
    "googlenet": 6,
    "resnet50": 7,
    "vgg16": 9,
    "densenet161": 12,
}

PLACEHOLDER_MODELS = ["resnet50", "vgg16", "googlenet", "densenet161"]

# Reuse the generic resource files that were already shown to work with glet.
TEMPLATE_RESOURCE_DIR = Path("/tmp/glet-mini")


@dataclass
class Case:
    name: str
    timeout_s: int
    task_fields: list[int]
    model_names: list[str]
    mem_mb: dict[str, int]
    batch_curve: dict[int, float]
    parts: list[int]
    use_parts: bool


def load_pi05_mean_rps() -> list[float]:
    d = json.loads(PI05_AUTOH.read_text())
    mean_h = float(d["mean_horizon"])
    return [30.0 / mean_h, 20.0 / mean_h, 10.0 / mean_h, 10.0 / mean_h]


def load_gr00t_mean_rps() -> list[float]:
    d = json.loads(GR00T_AUTOH.read_text())
    mean_h = float(d["mean_horizon"])
    return [30.0 / mean_h, 20.0 / mean_h, 10.0 / mean_h, 10.0 / mean_h]


def load_pi05_curve() -> dict[int, float]:
    d = json.loads(PI05_BATCH.read_text())
    curve = {int(x["batch_size"]): float(x["full_e2e"]["p50_ms"]) for x in d["results"]}
    last = max(curve)
    inc = max(0.5, curve[last] - curve[last - 1])
    for b in range(last + 1, 33):
        curve[b] = curve[b - 1] + inc
    return curve


def load_gr00t_curve() -> dict[int, float]:
    d = json.loads(GR00T_BATCH.read_text())
    curve = {int(x["batch_size"]): float(x["service_ms_for_scheduler"]) for x in d["results"]}
    last = max(curve)
    inc = max(0.5, curve[last] - curve[last - 1])
    for b in range(last + 1, 33):
        curve[b] = curve[b - 1] + inc
    return curve


def nearest_int_rates(rates: list[float]) -> list[int]:
    return [int(round(x)) for x in rates]


def scaled_int_rates(rates: list[float], scale: int) -> list[int]:
    return [max(1, int(round(x * scale))) for x in rates]


def make_task_csv(fields: list[int]) -> str:
    assert len(fields) % 3 == 0
    ntask = len(fields) // 3
    return f"{ntask}\n" + ",".join(str(x) for x in fields) + ","


def build_latency_csv(model_names: list[str], curve: dict[int, float], parts: list[int]) -> str:
    rows: list[str] = []
    profiled_batches = [1, 2, 4, 8, 16, 32]
    for model in model_names:
        for part in parts:
            for batch in profiled_batches:
                latency = curve[batch] * (100.0 / part)
                rows.append(f"{model},{part},{batch},{latency:.6f}")
    return "\n".join(rows) + "\n"


def prepare_resource_dir(case: Case) -> Path:
    workdir = Path(tempfile.mkdtemp(prefix=f"glet-{case.name}-", dir="/tmp"))
    (workdir / "4090").mkdir(parents=True, exist_ok=True)

    if TEMPLATE_RESOURCE_DIR.exists():
        for name in ["1_28_28.txt", "3_224_224.txt", "3_300_300.txt"]:
            shutil.copy2(TEMPLATE_RESOURCE_DIR / name, workdir / name)
        shutil.copy2(TEMPLATE_RESOURCE_DIR / "4090" / "INT_MODEL_CONSTANT.CSV", workdir / "4090" / "INT_MODEL_CONSTANT.CSV")
        shutil.copy2(TEMPLATE_RESOURCE_DIR / "4090" / "UTIL.CSV", workdir / "4090" / "UTIL.CSV")
    else:
        raise FileNotFoundError(f"Missing template resource dir: {TEMPLATE_RESOURCE_DIR}")

    (workdir / "4090" / "latency.csv").write_text(build_latency_csv(case.model_names, case.batch_curve, case.parts))

    device = {
        "device_specs": [
            {
                "type": "4090",
                "mem_mb": 24564,
                "latency_prof_file": "4090/latency.csv",
                "interference_const_file": "4090/INT_MODEL_CONSTANT.CSV",
                "interference_util_file": "4090/UTIL.CSV",
            }
        ]
    }
    sched = {
        "GPUs": [{"Type": "4090", "Num": 1}],
        "Max Model": len(case.model_names),
        "Part": 1 if case.use_parts else 0,
        "Latency Ratio": 1.1,
        "Interference": 0,
        "Avail_Parts": case.parts,
        "No Check": 1,
        "Incremental": 1,
    }
    full_mem = json.loads(GLET_MEM_CONFIG.read_text())
    for entry in full_mem["models"]:
        if entry["name"] in case.mem_mb:
            entry["mem"] = case.mem_mb[entry["name"]]

    (workdir / "device-config.json").write_text(json.dumps(device))
    (workdir / "sched-config.json").write_text(json.dumps(sched))
    (workdir / "mem-config.json").write_text(json.dumps(full_mem))
    (workdir / "proxy_config.json").write_text("{}")
    (workdir / "tasks.csv").write_text(make_task_csv(case.task_fields))
    return workdir


def run_glet_case(case: Case) -> dict[str, Any]:
    workdir = prepare_resource_dir(case)
    out = workdir / "ModelList.txt"
    cmd = [
        "timeout",
        f"{case.timeout_s}s",
        str(GLET_BIN),
        "--resource_dir",
        str(workdir),
        "--task_config",
        str(workdir / "tasks.csv"),
        "--sched_config",
        str(workdir / "sched-config.json"),
        "--output",
        str(out),
        "--mem_config",
        str(workdir / "mem-config.json"),
        "--device_config",
        str(workdir / "device-config.json"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stdout = res.stdout
    return {
        "exit_code": res.returncode,
        "timed_out": res.returncode == 124,
        "output_exists": out.exists(),
        "model_list": out.read_text() if out.exists() else "",
        "stdout_tail": stdout[-4000:],
        "workdir": str(workdir),
    }


def main() -> None:
    pi05_rates = load_pi05_mean_rps()
    gr00t_rates = load_gr00t_mean_rps()
    pi05_curve = load_pi05_curve()
    gr00t_curve = load_gr00t_curve()

    results: dict[str, Any] = {
        "environment": {
            "glet_root": str(GLET_ROOT),
            "glet_bin_exists": GLET_BIN.exists(),
            "template_resource_dir": str(TEMPLATE_RESOURCE_DIR),
        },
        "pi05_request_rate_mismatch": {
            "mean_request_rps": pi05_rates,
            "nearest_integer_rps": nearest_int_rates(pi05_rates),
            "scaled_x10_integer_rps": scaled_int_rates(pi05_rates, 10),
            "note": "glet offline scheduler only accepts integer request rates, while Pi0.5 AutoHorizon mean request rates are sub-1 RPS for the 10Hz robots.",
        },
        "gr00t_mean_request_rps": gr00t_rates,
        "cases": {},
    }

    sanity = Case(
        name="sanity_single_vgg16",
        timeout_s=15,
        task_fields=[FIXED_MODEL_IDS["vgg16"], 40, 1000],
        model_names=["vgg16"],
        mem_mb={"vgg16": 2920},
        batch_curve={1: 1.0, 2: 1.5, 4: 2.5, 8: 4.5, 16: 8.5, 32: 16.5},
        parts=[20, 40, 50, 60, 80, 100],
        use_parts=False,
    )
    # Expand the tiny sanity curve to 32 for code simplicity.
    last = 32
    sanity.batch_curve = {b: sanity.batch_curve.get(b, 16.5) for b in range(1, last + 1)}

    gr00t_two = Case(
        name="gr00t_two_model_gpulet",
        timeout_s=30,
        task_fields=[FIXED_MODEL_IDS["resnet50"], 6, 100, FIXED_MODEL_IDS["vgg16"], 4, 100],
        model_names=["resnet50", "vgg16"],
        mem_mb={"resnet50": 6268, "vgg16": 6268},
        batch_curve=gr00t_curve,
        parts=[50, 100],
        use_parts=True,
    )
    gr00t_four = Case(
        name="gr00t_four_model_gpulet",
        timeout_s=30,
        task_fields=[
            FIXED_MODEL_IDS["resnet50"], 6, 100,
            FIXED_MODEL_IDS["vgg16"], 4, 100,
            FIXED_MODEL_IDS["googlenet"], 2, 100,
            FIXED_MODEL_IDS["densenet161"], 2, 100,
        ],
        model_names=PLACEHOLDER_MODELS,
        mem_mb={
            "resnet50": 6268,
            "vgg16": 6268,
            "googlenet": 8767,
            "densenet161": 8767,
        },
        batch_curve=gr00t_curve,
        parts=[50, 100],
        use_parts=True,
    )
    pi05_scaled = Case(
        name="pi05_four_model_scaledx10_gpulet",
        timeout_s=30,
        task_fields=[
            FIXED_MODEL_IDS["resnet50"], scaled_int_rates(pi05_rates, 10)[0], 100,
            FIXED_MODEL_IDS["vgg16"], scaled_int_rates(pi05_rates, 10)[1], 100,
            FIXED_MODEL_IDS["googlenet"], scaled_int_rates(pi05_rates, 10)[2], 100,
            FIXED_MODEL_IDS["densenet161"], scaled_int_rates(pi05_rates, 10)[3], 100,
        ],
        model_names=PLACEHOLDER_MODELS,
        mem_mb={name: 7665 for name in PLACEHOLDER_MODELS},
        batch_curve=pi05_curve,
        parts=[50, 100],
        use_parts=True,
    )

    for case in [sanity, gr00t_two, gr00t_four, pi05_scaled]:
        results["cases"][case.name] = run_glet_case(case)

    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
