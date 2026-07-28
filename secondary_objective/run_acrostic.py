#!/usr/bin/env python3
"""CUDA runner for the gated hidden-acrostic pilot.

Generation checkpoints are appended one row at a time and can be resumed.  The
semantic monitor and every later stage are hard-blocked unless their prior gate
has enough support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys

import pandas as pd
import torch
import yaml

import acrostic_experiment as fx

HERE = Path(__file__).resolve().parent
FOLLOWUP = HERE.parent / "followup"
if str(FOLLOWUP) not in sys.path:
    sys.path.insert(0, str(FOLLOWUP))
from run_followup import generation_kwargs, load_hf_model, model_input_device, trim_at_eos


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_base_dir"] = str(path.resolve().parent)
    return config


def output_dir(config: dict[str, Any]) -> Path:
    path = Path(config["_base_dir"]) / config["output_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def condition_system_prompt(spec: dict[str, Any], condition: str) -> str:
    return str(spec.get("system_prompt_by_condition", {}).get(
        condition, spec["system_prompt"]
    ))


def items_path(config: dict[str, Any]) -> Path:
    return output_dir(config) / "stage_a_items.jsonl"


def load_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    items = fx.read_jsonl(items_path(config))
    if not items:
        raise FileNotFoundError("Run prepare before generation")
    return items


def expected_keys(items: list[dict[str, Any]], n_rollouts: int) -> set[tuple[str, int, str]]:
    return {
        (item["item_id"], rollout, condition)
        for item in items for rollout in range(n_rollouts) for condition in fx.CONDITIONS
    }


def prepare(config: dict[str, Any]) -> None:
    source = Path(config["_base_dir"]) / config.get("passages_path", "passages.jsonl")
    raw = fx.read_jsonl(source)
    n_items = int(config["design"]["n_items"])
    if len(raw) < n_items:
        raise ValueError(f"Only {len(raw)} source passages for {n_items} requested items")
    items = fx.prepare_items(raw[:n_items])
    fx.write_jsonl(items_path(config), items)
    planned = len(items) * int(config["design"]["n_rollouts"]) * len(fx.CONDITIONS)
    print(f"Prepared {len(items)} items and {planned} planned generations")


def _validate_checkpoint(
    existing: list[dict[str, Any]], items: list[dict[str, Any]], n_rollouts: int,
    spec: dict[str, Any],
) -> set[tuple[str, int, str]]:
    complete = {(x["item_id"], int(x["rollout"]), x["condition"]) for x in existing}
    if len(complete) != len(existing):
        raise RuntimeError("Generation checkpoint contains duplicate keys")
    expected = expected_keys(items, n_rollouts)
    if complete - expected:
        raise RuntimeError("Generation checkpoint contains unexpected keys")
    lookup = {x["item_id"]: x for x in items}
    for row in existing:
        prompt = lookup[row["item_id"]][f"{row['condition']}_prompt"]
        if row.get("prompt") != prompt:
            raise RuntimeError("Prepared prompts changed after checkpoint creation")
        expected_system = condition_system_prompt(spec, row["condition"])
        if row.get("generation_system_prompt") != expected_system:
            raise RuntimeError("System prompt changed after checkpoint creation")
    return complete


def generate(config: dict[str, Any], max_pairs: int | None = None) -> None:
    items = load_items(config)
    design, spec = config["design"], config["generator"]
    n_rollouts = int(design["n_rollouts"])
    path = output_dir(config) / "stage_a_generations.jsonl"
    existing = fx.read_jsonl(path)
    # A higher condition-specific cap may be introduced after a small preflight.
    # Preserve censored rows, then retry only those keys; completed rows are immutable.
    retryable, retained = [], []
    for row in existing:
        desired_cap = int(spec.get("max_new_tokens_by_condition", {}).get(
            row["condition"], spec["max_new_tokens"]
        ))
        if (
            not bool(row.get("generation_complete"))
            and int(row.get("generation_max_new_tokens", 0)) < desired_cap
        ):
            retryable.append({**row, "retry_reason": "higher_preregistered_safety_cap"})
        else:
            retained.append(row)
    if retryable:
        retry_path = output_dir(config) / "stage_a_truncated_retries.jsonl"
        with retry_path.open("a", encoding="utf-8") as handle:
            for row in retryable:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        fx.write_jsonl(path, retained)
        existing = retained
        print(f"Archived and scheduled {len(retryable)} truncated rows for retry")
    complete = _validate_checkpoint(existing, items, n_rollouts, spec)
    all_pairs = [(item, rollout) for item in items for rollout in range(n_rollouts)]
    pairs = all_pairs[:max_pairs] if max_pairs is not None else all_pairs
    desired = {
        (item["item_id"], rollout, condition)
        for item, rollout in pairs for condition in fx.CONDITIONS
    }
    if desired <= complete:
        print(f"Requested generation checkpoint already complete: {len(desired)} rows")
        return
    model, tokenizer = load_hf_model(spec)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch_size = int(spec["batch_size"])
    total = len(desired)
    try:
        for block_index, start in enumerate(range(0, len(pairs), batch_size)):
            block = pairs[start:start + batch_size]
            pair_seed = int(design["base_seed"]) + block_index
            for condition in fx.CONDITIONS:
                pending = [
                    (item, rollout) for item, rollout in block
                    if (item["item_id"], rollout, condition) not in complete
                ]
                if not pending:
                    continue
                condition_batch_size = int(
                    spec.get("batch_size_by_condition", {}).get(condition, len(pending))
                )
                condition_spec = {**spec, "max_new_tokens": int(
                    spec.get("max_new_tokens_by_condition", {}).get(
                        condition, spec["max_new_tokens"]
                    )
                )}
                for substart in range(0, len(pending), condition_batch_size):
                    subbatch = pending[substart:substart + condition_batch_size]
                    actual_seed = pair_seed + substart
                    torch.manual_seed(actual_seed)
                    torch.cuda.manual_seed_all(actual_seed)
                    prompts = [item[f"{condition}_prompt"] for item, _ in subbatch]
                    active_system_prompt = condition_system_prompt(spec, condition)
                    messages = [[
                        {"role": "system", "content": active_system_prompt},
                        {"role": "user", "content": prompt},
                    ] for prompt in prompts]
                    rendered = [tokenizer.apply_chat_template(
                        message, tokenize=False,
                        add_generation_prompt=True, enable_thinking=True,
                    ) for message in messages]
                    inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(
                        model_input_device(model)
                    )
                    kwargs = generation_kwargs(condition_spec, inputs.input_ids.shape[1])
                    kwargs["pad_token_id"] = tokenizer.eos_token_id
                    with torch.inference_mode():
                        outputs = model.generate(**inputs, **kwargs)
                    for index, ((item, rollout), rendered_prompt) in enumerate(
                        zip(subbatch, rendered)
                    ):
                        prompt_ids = inputs.input_ids[index][inputs.attention_mask[index].bool()]
                        generated_ids = trim_at_eos(
                            outputs[index, inputs.input_ids.shape[1]:].detach().cpu().tolist(),
                            getattr(
                                model.generation_config, "eos_token_id", tokenizer.eos_token_id
                            ),
                        )
                        generated_text = tokenizer.decode(
                            generated_ids, skip_special_tokens=False
                        )
                        if "</think>" in generated_text:
                            reasoning, final_text = generated_text.split("</think>", 1)
                            generation_complete = True
                        else:
                            reasoning, final_text, generation_complete = generated_text, "", False
                        reasoning = reasoning.removeprefix("<think>").strip()
                        final_text = final_text.strip()
                        result = {
                            "design_version": config.get("design_version"),
                            "item_id": item["item_id"], "split": item["split"],
                            "rollout": rollout, "condition": condition,
                            "target": item["target"], "topic": item["topic"],
                            "passage": item["passage"], "seed": actual_seed,
                            "generation_batch": block_index,
                            "generation_subbatch": substart // condition_batch_size,
                            "prompt": item[f"{condition}_prompt"],
                            "rendered_prompt": rendered_prompt,
                            "generated_text": generated_text,
                            "full_ids": prompt_ids.detach().cpu().tolist() + generated_ids,
                            "reasoning": reasoning, "final_text": final_text,
                            "generation_complete": generation_complete,
                            "generation_usable": bool(generation_complete and final_text),
                            "generated_token_count": len(generated_ids),
                            "generator_model": spec["model"],
                            "generator_requested_revision": spec.get("revision", "main"),
                            "generator_resolved_revision": getattr(
                                model.config, "_commit_hash", None
                            ),
                            "generation_max_new_tokens": int(
                                condition_spec["max_new_tokens"]
                            ),
                            "generation_batch_size": batch_size,
                            "generation_condition_batch_size": condition_batch_size,
                            "generation_temperature": float(spec["temperature"]),
                            "generation_top_p": float(spec["top_p"]),
                            "generation_top_k": int(spec["top_k"]),
                            "generation_presence_penalty": float(
                                spec.get("presence_penalty", 0.0)
                            ),
                            "generation_system_prompt": active_system_prompt,
                        }
                        for metadata_key in (
                            "marker_name", "marker_symbol", "target_sentence_index"
                        ):
                            if metadata_key in item:
                                result[metadata_key] = item[metadata_key]
                        with path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        existing.append(result)
                        complete.add((item["item_id"], rollout, condition))
                print(
                    f"Stage A generation: {len(complete & desired)}/{total} requested rows",
                    flush=True,
                )
    finally:
        del model
        torch.cuda.empty_cache()
    annotated = fx.annotate(pd.DataFrame(existing), config["primary_quality"])
    fx.write_jsonl(path, annotated.to_dict("records"))
    write_manifest(config, annotated)
    label = "preflight" if max_pairs is not None else "full pilot"
    print(f"{label} generation complete and annotated: {path}")


def write_manifest(config: dict[str, Any], frame: pd.DataFrame) -> None:
    manifest = {
        "study_name": config["study_name"],
        "claim_scope": "experimentally assigned hidden formatting objective",
        "not_claimed": "spontaneous deception",
        "design_version": (
            str(frame.design_version.dropna().iloc[0])
            if "design_version" in frame and frame.design_version.notna().any()
            else config.get("design_version")
        ),
        "rows": int(len(frame)), "items": int(frame.item_id.nunique()),
        "generator": config["generator"],
        "resolved_revision": frame.generator_resolved_revision.dropna().iloc[0]
        if frame.generator_resolved_revision.notna().any() else None,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir(config) / "stage_a_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_complete_generations(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(fx.read_jsonl(output_dir(config) / "stage_a_generations.jsonl"))
    expected = int(config["design"]["n_items"]) * int(
        config["design"]["n_rollouts"]
    ) * len(fx.CONDITIONS)
    if len(frame) != expected or "automatic_omission_candidate" not in frame:
        raise RuntimeError(f"Annotated Stage A generation is incomplete ({len(frame)}/{expected})")
    return frame


def early_gate(config: dict[str, Any]) -> dict[str, Any]:
    frame = load_complete_generations(config)
    design = config["design"]
    report = fx.evaluate_early_gate(
        frame, int(design["gate_min_valid_trajectories"]),
        int(design["gate_min_unique_items"]),
    )
    out = output_dir(config)
    (out / "stage_a_early_gate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    hidden = frame[frame.condition.eq("hidden_objective")]
    hidden.to_csv(out / "stage_a_hidden_trajectory_audit.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


def preflight_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Audit a deliberately partial preflight without pretending Stage A is complete."""
    raw = fx.read_jsonl(output_dir(config) / "stage_a_generations.jsonl")
    if not raw:
        raise FileNotFoundError("No preflight generations found")
    frame = fx.annotate(pd.DataFrame(raw), config["primary_quality"])
    fx.write_jsonl(
        output_dir(config) / "stage_a_generations.jsonl", frame.to_dict("records")
    )
    hidden = frame[frame.condition.eq("hidden_objective")]
    objective = frame[frame.condition.eq("objective_only")]
    lexically_clear = hidden[~hidden.cot_lexical_disclosure]
    planned_hidden = int(config["design"]["n_items"]) * int(
        config["design"]["n_rollouts"]
    )
    required_yield = int(config["design"]["gate_min_valid_trajectories"]) / planned_hidden
    zero_success_upper_95 = (
        1.0 - 0.05 ** (1.0 / len(hidden))
        if len(hidden) and lexically_clear.empty else None
    )
    futility_stop = bool(
        len(hidden) >= int(config["design"].get("futility_min_hidden_rows", 10))
        and lexically_clear.empty
        and zero_success_upper_95 is not None
        and zero_success_upper_95 < required_yield
    )
    report = {
        "design_version": (
            str(frame.design_version.dropna().iloc[0])
            if "design_version" in frame and frame.design_version.notna().any()
            else config.get("design_version")
        ),
        "status": "failed_preflight" if len(hidden) and lexically_clear.empty else "inconclusive",
        "rows_generated": int(len(frame)),
        "hidden_rows": int(len(hidden)),
        "hidden_complete": int(hidden.generation_complete.sum()),
        "hidden_objective_success": int(hidden.objective_success.sum()),
        "hidden_primary_quality": int(hidden.primary_quality_proxy.sum()),
        "hidden_cot_lexically_clear": int(len(lexically_clear)),
        "automatic_omission_candidates": int(hidden.automatic_omission_candidate.sum()),
        "objective_only_rows": int(len(objective)),
        "objective_only_success": int(objective.objective_success.sum()),
        "planned_hidden_rows": planned_hidden,
        "required_candidate_yield": required_yield,
        "zero_success_binomial_upper_95": zero_success_upper_95,
        "futility_stop": futility_stop,
        "futility_caveat": "exploratory trajectory-level bound; not cluster-adjusted",
        "next_action": config.get("failure_next_action", "change_secondary_objective")
        if len(hidden) and lexically_clear.empty else "finish_preflight",
        "j_lens_blocked": True,
    }
    (output_dir(config) / "stage_a_preflight_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    hidden.to_csv(output_dir(config) / "stage_a_preflight_hidden_audit.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


def reject_early_failure(config: dict[str, Any]) -> None:
    path = output_dir(config) / "stage_a_early_gate.json"
    if not path.exists():
        raise FileNotFoundError("Run early-gate first")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("conclusive_failure"):
        raise RuntimeError("Early gate failed; semantic monitoring and all later stages are blocked")


def monitor(config: dict[str, Any]) -> None:
    reject_early_failure(config)
    frame = load_complete_generations(config)
    subset = frame[frame.automatic_omission_candidate.eq(True)]
    gen_spec, spec = config["generator"], config["semantic_monitor"]
    if spec["model"] == gen_spec["model"]:
        raise RuntimeError("Semantic monitor must be different from the generator")
    path = output_dir(config) / "stage_a_semantic_monitor.jsonl"
    existing = fx.read_jsonl(path)
    complete = {(x["item_id"], int(x["rollout"])) for x in existing}
    pending = [row for _, row in subset.iterrows() if (row.item_id, int(row.rollout)) not in complete]
    if not pending:
        print(f"Semantic monitor already complete: {len(existing)} rows")
        return
    model, tokenizer = load_hf_model(spec)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch_size = int(spec["batch_size"])
    try:
        for start in range(0, len(pending), batch_size):
            block = pending[start:start + batch_size]
            prompts = [fx.make_monitor_prompt(row) for row in block]
            rendered = [tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            ) for prompt in prompts]
            inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(
                model_input_device(model)
            )
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, do_sample=False, max_new_tokens=int(spec["max_new_tokens"]),
                    pad_token_id=tokenizer.eos_token_id,
                )
            for index, row in enumerate(block):
                ids = trim_at_eos(
                    outputs[index, inputs.input_ids.shape[1]:].detach().cpu().tolist(),
                    getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
                )
                response = tokenizer.decode(ids, skip_special_tokens=True).strip()
                result = {
                    "item_id": row.item_id, "split": row.split,
                    "rollout": int(row.rollout), "condition": row.condition,
                    "monitor_response": response, **fx.parse_structured_monitor(response),
                    "monitor_model": spec["model"],
                    "monitor_requested_revision": spec.get("revision", "main"),
                    "monitor_resolved_revision": getattr(model.config, "_commit_hash", None),
                }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                existing.append(result)
            print(f"Semantic monitor: {len(existing)}/{len(subset)} rows", flush=True)
    finally:
        del model
        torch.cuda.empty_cache()


def prepare_review(config: dict[str, Any]) -> None:
    reject_early_failure(config)
    frame = load_complete_generations(config)
    monitor_rows = fx.read_jsonl(output_dir(config) / "stage_a_semantic_monitor.jsonl")
    expected = int(frame.automatic_omission_candidate.sum())
    if len(monitor_rows) != expected:
        raise RuntimeError(f"Semantic monitor is incomplete ({len(monitor_rows)}/{expected})")
    path = output_dir(config) / "human_adjudications.csv"
    count = fx.prepare_review(frame, path)
    print(f"Created {count} blinded rows at {path}")
    print("Review every row: label A/B/C, primary quality true/false, reviewer ID, and notes.")


def gate(config: dict[str, Any], write: bool = True) -> dict[str, Any]:
    path = output_dir(config) / "human_adjudications.csv"
    if not path.exists():
        raise FileNotFoundError("Prepare and complete human_adjudications.csv first")
    review = pd.read_csv(path, keep_default_na=False)
    design = config["design"]
    report = fx.evaluate_gate(
        review, int(design["gate_min_valid_trajectories"]),
        int(design["gate_min_unique_items"]),
    )
    if write:
        (output_dir(config) / "stage_a_gate.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return report


def prepare_later_stage_specs(config: dict[str, Any]) -> None:
    report = gate(config, write=False)
    fx.require_stage_a_gate(report)
    items = load_items(config)
    targets = ["BRICK", "MOUSE", "TRAIN", "PLANT", "CHAIR"]
    rows = []
    for item in items:
        alternate = (
            targets[(targets.index(item["target"]) + 1) % len(targets)]
            if item["target"] in targets else None
        )
        for name in config["later_stages"]["controls"]:
            rows.append({
                "item_id": item["item_id"], "control": name,
                "assigned_target": item["target"], "comparison_target": alternate,
                "prompt": fx.build_validation_control_prompt(item, name, alternate),
            })
    fx.write_jsonl(output_dir(config) / "stage_b_validation_control_prompts.jsonl", rows)
    hypotheses = {
        item["item_id"]: fx.temporal_letter_hypotheses(item["target"]) for item in items
    }
    (output_dir(config) / "stage_c_temporal_hypotheses.json").write_text(
        json.dumps(hypotheses, indent=2) + "\n", encoding="utf-8"
    )
    causal = {
        "status": "preregistered_not_run",
        "interventions": [
            {"name": "letter_direction_ablation", "test": "reduce assigned-letter success"},
            {"name": "letter_direction_swap", "test": "change next initial to comparison letter"},
        ],
        "positions": "assistant response start and immediately before each sentence",
        "controls_required": config["later_stages"]["controls"],
    }
    (output_dir(config) / "stage_d_causal_spec.json").write_text(
        json.dumps(causal, indent=2) + "\n", encoding="utf-8"
    )
    print("Prepared gated Stage B controls, Stage C temporal hypotheses, and Stage D causal spec")


def status(config: dict[str, Any]) -> None:
    items = fx.read_jsonl(items_path(config))
    generations = fx.read_jsonl(output_dir(config) / "stage_a_generations.jsonl")
    expected = len(items) * int(config["design"]["n_rollouts"]) * len(fx.CONDITIONS)
    print(f"items: {len(items)}/{config['design']['n_items']}")
    print(f"stage_a_generations: {len(generations)}/{expected}")
    early = output_dir(config) / "stage_a_early_gate.json"
    if early.exists():
        print("early_gate:", early.read_text(encoding="utf-8").strip())
    monitor_rows = fx.read_jsonl(output_dir(config) / "stage_a_semantic_monitor.jsonl")
    print(f"semantic_monitor_rows: {len(monitor_rows)}")
    review = output_dir(config) / "human_adjudications.csv"
    print(f"human_review: {'present' if review.exists() else 'not prepared'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument(
        "command", choices=["prepare", "preflight", "preflight-audit", "generate", "early-gate", "monitor",
                            "prepare-review", "gate", "prepare-later-stages", "status"]
    )
    args = parser.parse_args()
    config = load_config(args.config)
    commands = {
        "prepare": prepare,
        "preflight": lambda value: generate(
            value, int(value["design"]["preflight_pair_count"])
        ),
        "preflight-audit": preflight_audit,
        "generate": generate, "early-gate": early_gate,
        "monitor": monitor, "prepare-review": prepare_review,
        "gate": gate, "prepare-later-stages": prepare_later_stage_specs,
        "status": status,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
