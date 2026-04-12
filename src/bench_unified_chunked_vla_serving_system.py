#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/VLAServing")
RESULTS = ROOT / "results"
OUT = RESULTS / "unified_chunked_vla_serving_system_20260412.json"

PI05_PREFETCH = RESULTS / "pi05_four_model_residency_prefetch_system_20260406.json"
PI05_AUTOH = RESULTS / "pi05_frequency_aware_page_prefetch_autohorizon_20260406.json"
GR00T_BATCH = RESULTS / "gr00t_shared_prefix_phase_lock_batch_mps_20260412.json"
GR00T_FAIR = RESULTS / "gr00t_batch_only_fair_admission_20260412.json"
GR00T_PHASE = RESULTS / "gr00t_phase_correction_tradeoff_quick_20260412.json"


@dataclass(frozen=True)
class GeneralInterface:
    field: str
    meaning: str


@dataclass(frozen=True)
class BackendChoice:
    name: str
    when_to_use: str
    control_knobs: list[str]
    required_model_properties: list[str]
    disallowed_assumptions: list[str]


@dataclass(frozen=True)
class FamilyBlueprint:
    family: str
    chunk_size: int
    default_backend: str
    scheduler: str
    admission: str
    memory_strategy: str
    phase_control: str
    batching: str
    mps_policy: str
    validated_reference: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def build_general_interfaces() -> list[GeneralInterface]:
    return [
        GeneralInterface("chunk_size", "Number of actions produced by one VLA inference."),
        GeneralInterface("horizon_process", "Distribution or controller that decides how many actions are consumed before the next request."),
        GeneralInterface("request_period_hz", "Robot control/request frequency."),
        GeneralInterface("request_window", "Time-action window in which a new inference may legally happen."),
        GeneralInterface("latency_model", "Per-request inference cost or batch latency curve."),
        GeneralInterface("memory_relation", "How fine-tuned models relate to the base model: full-copy, shared-prefix, or exact-delta."),
        GeneralInterface("batch_domain", "Whether requests may batch across all models, same-model only, or not at all."),
        GeneralInterface("phase_correction_budget", "How aggressively the runtime may pull inference earlier to align future requests."),
        GeneralInterface("admission_policy", "How new robots are admitted while preserving deadline and fairness constraints."),
    ]


def build_backends() -> list[BackendChoice]:
    return [
        BackendChoice(
            name="ExactDeltaPrefetch",
            when_to_use="Fine-tunes differ mainly by exact delta pages and the GPU cannot keep all full weights resident.",
            control_knobs=[
                "resident_fraction_per_model",
                "next_request_prefetch_window_ms",
                "frequency_aware_residency",
                "decode_apply_overlap",
                "shared_shell_count",
            ],
            required_model_properties=[
                "Exact or lossless delta reconstruction",
                "Predictable next-request timing from chunked control",
                "Transfer/decode cost materially larger than compute",
            ],
            disallowed_assumptions=[
                "Do not assume same-model batching is the dominant win",
                "Do not assume every fine-tune fits fully resident",
            ],
        ),
        BackendChoice(
            name="SharedPrefixPhaseBatch",
            when_to_use="Fine-tunes share a resident prefix or can all remain resident, and same-model batch latency scales well.",
            control_knobs=[
                "phase_correction_budget_actions",
                "same_model_batch_size_cap",
                "target_phase_per_model",
                "fair_quota_per_model",
            ],
            required_model_properties=[
                "Same architecture across fine-tunes",
                "Same-model batch curve available",
                "Legal early replan window inside the chunk",
            ],
            disallowed_assumptions=[
                "Do not assume MPS always increases stable capacity",
                "Do not assume greedy batch-first admission is fair",
            ],
        ),
        BackendChoice(
            name="HybridChunkedRuntime",
            when_to_use="A new chunked-action model family only partially satisfies the assumptions of the two specialized backends.",
            control_knobs=[
                "backend_selection_rule",
                "per-model residency tier",
                "phase_correction_budget_actions",
                "quota_fairness",
            ],
            required_model_properties=[
                "Chunked action outputs",
                "Predictable re-request semantics",
                "At least one measurable latency model",
            ],
            disallowed_assumptions=[
                "Do not hardcode Pi05 or GR00T specific names",
                "Do not require a particular backbone architecture",
            ],
        ),
    ]


def build_pi05_blueprint() -> FamilyBlueprint:
    prefetch = load_json(PI05_PREFETCH)
    autoh = load_json(PI05_AUTOH)
    best = prefetch["best"]
    return FamilyBlueprint(
        family="pi05",
        chunk_size=50,
        default_backend="ExactDeltaPrefetch",
        scheduler="single active shell + predictive prefetch + decode/apply overlap",
        admission="frequency-aware residency with predictive windows; fairness can be layered on top",
        memory_strategy="three active shells resident; low-frequency models keep compressed/exact-delta pages resident by fraction",
        phase_control="use AutoHorizon future request times; phase shifting is weak compared with memory scheduling and not the primary lever",
        batching="not the primary optimization path for Pi05 fine-tuned families in this workspace",
        mps_policy="disabled by default; not part of the validated Pi05 fast path",
        validated_reference={
            "reference_file": str(PI05_PREFETCH),
            "validated_models": 4,
            "robot_period_ms": 500.0,
            "deadline_passes_ms": [45, 50, 55, 60, 65, 70, 80, 100],
            "effective_swap_ms": best["effective_swap_ms"],
            "gpu_memory_total_gb": best["gpu_memory_estimate_gb"]["total_estimated_gb"],
            "fits_under_24gb": best["gpu_memory_estimate_gb"]["fits_under_24gb"],
            "service_backlog_p95_ms": best["backlog_ms"]["p95_ms"],
            "per_model_e2e_p95_ms": {
                model: stats["e2e_ms"]["p95_ms"] for model, stats in best["per_model"].items()
            },
            "autohorizon_admission_reference": {
                "reference_file": str(PI05_AUTOH),
                "mean_admitted_total": autoh["admission"]["summary"]["mean_admitted_total"],
                "admitted_histogram": autoh["admission"]["summary"]["admitted_histogram"],
            },
        },
    )


def build_gr00t_blueprint() -> FamilyBlueprint:
    batch = load_json(GR00T_BATCH)
    fair = load_json(GR00T_FAIR)
    phase = load_json(GR00T_PHASE)
    batch_only = batch["scenarios"]["4x_per_model"]["batch_only"]["best"]["metrics"]
    batch_plus_mps = batch["scenarios"]["4x_per_model"]["batch_plus_mps"]["best"]["metrics"]
    quota_fair = fair["quota_fair"]["summary"]
    phase_frontier = phase["frontier"]
    return FamilyBlueprint(
        family="gr00t_n1d6",
        chunk_size=16,
        default_backend="SharedPrefixPhaseBatch",
        scheduler="same-model phase-lock batching with predictive correction inside the legal chunk window",
        admission="quota-fair admission with per-model target lanes; do not use greedy batch-first as the production default",
        memory_strategy="shared-prefix resident copies; all fine-tunes remain resident instead of swapping whole models",
        phase_control="enabled by default; budget 4 actions is the knee point, budget >4 adds no value in current evidence",
        batching="same-model only; target full batch cohorts per model",
        mps_policy="optional only; disabled by default because validated stable capacity does not improve over batch-only",
        validated_reference={
            "reference_file": str(GR00T_BATCH),
            "validated_models": 4,
            "stable_robot_count_batch_only": 16,
            "stable_robot_count_batch_plus_mps": 16,
            "batch_only_p95_ms_at_16": batch_only["mean_request_to_result_p95_ms"],
            "batch_plus_mps_p95_ms_at_16": batch_plus_mps["mean_request_to_result_p95_ms"],
            "batch_only_mean_batch_size_at_16": batch_only["mean_batch_size"],
            "batch_plus_mps_mean_batch_size_at_16": batch_plus_mps["mean_batch_size"],
            "fair_admission_reference": {
                "reference_file": str(GR00T_FAIR),
                "mean_final_robot_count": quota_fair["mean_final_robot_count"],
                "final_count_gap": quota_fair["final_count_gap"],
                "jain_final_count": quota_fair["jain_final_count"],
                "mean_p95_ms": quota_fair["mean_p95_ms"],
            },
            "phase_budget_frontier": phase_frontier,
        },
    )


def build_selection_rule() -> dict[str, Any]:
    return {
        "goal": "Choose a serving backend for any chunked-action VLA family without hardcoding Pi05 or GR00T specific assumptions.",
        "decision_tree": [
            {
                "if": "all fine-tunes can remain resident or share a fully resident prefix, and same-model batch latency is known",
                "then": "use SharedPrefixPhaseBatch",
            },
            {
                "if": "fine-tunes are reconstructable by exact/lossless deltas and whole-model residency is too expensive",
                "then": "use ExactDeltaPrefetch",
            },
            {
                "if": "neither assumption fully holds",
                "then": "use HybridChunkedRuntime and learn the backend choice per model family from measured latency + memory artifacts",
            },
        ],
        "required_measurements_for_a_new_family": [
            "single-request p50/p95 latency",
            "same-model batch curve, if batching is supported",
            "full-shell memory footprint",
            "exact-delta or shared-prefix memory footprint, if available",
            "AutoHorizon or fixed execution-horizon process",
            "legal early-replan window inside a chunk",
        ],
    }


def build_runtime_components() -> dict[str, Any]:
    return {
        "timeline_predictor": {
            "role": "Maintain the next-request timeline for every robot from its chunk horizon process.",
            "general_to_chunked_models": True,
        },
        "memory_state_manager": {
            "role": "Own resident shells/prefixes/delta pages and issue prefetch/decode/apply commands.",
            "backends": ["ExactDeltaPrefetch", "SharedPrefixPhaseBatch", "HybridChunkedRuntime"],
        },
        "compute_scheduler": {
            "role": "Schedule inference work so that request-to-result latency stays under the target deadline.",
            "submodes": [
                "single-queue shell reuse",
                "same-model batch waves",
                "optional MPS waves only when they do not reduce stable capacity",
            ],
        },
        "phase_controller": {
            "role": "Use legal early replan inside the chunk to lock future requests into better cohorts.",
            "note": "Mandatory for SharedPrefixPhaseBatch; optional/weak for ExactDeltaPrefetch families such as Pi05.",
        },
        "admission_controller": {
            "role": "Admit robots only when the predicted steady-state remains stable, deadline-safe, and fair.",
            "policy_defaults": {
                "gr00t_n1d6": "quota_fair",
                "pi05": "frequency_aware_prefetch",
            },
        },
    }


def build_payload() -> dict[str, Any]:
    pi05 = build_pi05_blueprint()
    gr00t = build_gr00t_blueprint()
    return {
        "system_name": "Unified Chunked VLA Serving System",
        "scope": "General serving architecture for chunked-action VLA families with predictive request timelines.",
        "general_interfaces": [asdict(item) for item in build_general_interfaces()],
        "backends": [asdict(item) for item in build_backends()],
        "selection_rule": build_selection_rule(),
        "runtime_components": build_runtime_components(),
        "families": {
            "pi05": asdict(pi05),
            "gr00t_n1d6": asdict(gr00t),
        },
        "unification_summary": {
            "common_primitives": [
                "future request timeline from chunk horizons",
                "deadline-aware admission",
                "family-specific memory backend behind a shared coordinator",
                "phase-aware scheduling when early replan is legal",
            ],
            "family_specific_specialization": {
                "pi05": "memory-bound exact-delta residency/prefetch backend",
                "gr00t_n1d6": "shared-prefix same-model phase-lock batch backend",
            },
            "general_claim": "The system generalizes to chunked-action VLA models by abstracting over horizon process, latency model, and fine-tune memory relation rather than hardcoding one model architecture.",
        },
    }


def main():
    payload = build_payload()
    OUT.write_text(json.dumps(payload, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
