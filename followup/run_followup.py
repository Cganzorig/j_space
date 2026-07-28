#!/usr/bin/env python3
"""Command-line runner for the gated phenomenon-first follow-up.

Run ``python run_followup.py --help`` from this directory. Long-running commands
write one JSONL row at a time and resume safely from existing checkpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
import yaml

import followup_experiment as fx


HERE = Path(__file__).resolve().parent
J_SPACE = HERE.parent
if str(J_SPACE) not in sys.path:
    sys.path.insert(0, str(J_SPACE))


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path.resolve())
    config["_base_dir"] = str(path.resolve().parent)
    return config


def resolve_path(config: dict[str, Any], value: str) -> Path:
    return (Path(config["_base_dir"]) / value).resolve()


def output_dir(config: dict[str, Any]) -> Path:
    path = resolve_path(config, config["output_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def reject_conclusive_early_failure(config: dict[str, Any]) -> None:
    path = output_dir(config) / "stage_a_early_gate.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("conclusive_failure"):
        raise RuntimeError(
            "This answer-cue pilot conclusively failed the Stage A upper-bound gate "
            f"({report.get('behavior_success_trajectories')} possible vs "
            f"{report.get('minimum_trajectories')} required). Semantic monitoring and "
            "Stage B are blocked; switch to the hidden-secondary-objective task."
        )


def load_items(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = output_dir(config) / "stage_a_items.jsonl"
    rows = fx.read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"Run prepare first; no items at {path}")
    return rows


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; choose {sorted(mapping)}")
    return mapping[name]


def load_hf_model(spec: dict[str, Any]):
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. Model generation/scoring was not started.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec["model"], revision=spec.get("revision", "main"),
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        spec["model"], revision=spec.get("revision", "main"),
        dtype=dtype_from_name(spec.get("dtype", "bfloat16")), device_map="auto",
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
    )
    model.eval()
    return model, tokenizer


class GeneratedPresencePenalty:
    """Transformers-compatible output-token presence penalty."""

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        generated = input_ids[:, self.prompt_length:]
        if generated.numel() == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated, True)
        return scores - seen.to(scores.dtype) * self.penalty


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def trim_at_eos(token_ids: list[int], eos: int | list[int] | None) -> list[int]:
    eos_set = {eos} if isinstance(eos, int) else set(eos or [])
    for index, token_id in enumerate(token_ids):
        if token_id in eos_set:
            return token_ids[: index + 1]
    return token_ids


def generation_kwargs(spec: dict[str, Any], prompt_length: int) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": int(spec["max_new_tokens"]),
        "do_sample": True,
        "temperature": float(spec["temperature"]),
        "top_p": float(spec["top_p"]),
        "top_k": int(spec["top_k"]),
    }
    penalty = float(spec.get("presence_penalty", 0.0))
    if penalty:
        kwargs["logits_processor"] = [GeneratedPresencePenalty(penalty, prompt_length)]
    return kwargs


def prepare(config: dict[str, Any]) -> None:
    dataset_path = resolve_path(config, config["dataset_path"])
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    design = config["design"]
    if tuple(design["conditions"]) != fx.ALL_CONDITIONS:
        raise ValueError(f"Conditions must remain {fx.ALL_CONDITIONS}")
    items = fx.prepare_items(
        records, int(design["n_items"]), int(design["n_development_items"])
    )
    path = output_dir(config) / "stage_a_items.jsonl"
    fx.write_jsonl(path, items)
    expected = len(items) * int(design["n_rollouts"]) * len(fx.ALL_CONDITIONS)
    print(f"Prepared {len(items)} items and {expected} planned generations at {path}")


def generate_stage_a(config: dict[str, Any]) -> None:
    items = load_items(config)
    design, spec = config["design"], config["generator"]
    path = output_dir(config) / "stage_a_generations.jsonl"
    existing = fx.read_jsonl(path)
    completed = {(x["item_id"], int(x["rollout"]), x["condition"]) for x in existing}
    expected = fx.expected_generation_keys(items, int(design["n_rollouts"]))
    if len(existing) != len(completed):
        raise RuntimeError("Generation checkpoint contains duplicate keys")
    unknown = completed - expected
    if unknown:
        raise RuntimeError(f"Checkpoint has unexpected keys: {list(unknown)[:3]}")
    item_lookup = {item["item_id"]: item for item in items}
    for row in existing:
        expected_prompt = item_lookup[row["item_id"]][f"{row['condition']}_prompt"]
        if row.get("prompt") != expected_prompt:
            raise RuntimeError(
                "A prepared prompt changed after generation began. Preserve the old "
                "checkpoint/config or start a new output directory."
            )
    if completed == expected:
        print(f"Generation checkpoint already complete: {len(completed)} rows")
        return
    model, tokenizer = load_hf_model(spec)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    pair_specs = [
        (item, rollout) for item in items for rollout in range(int(design["n_rollouts"]))
    ]
    batch_size = int(spec["batch_size"])
    base_seed = int(design["base_seed"])
    total = len(expected)
    try:
        for block_index, start in enumerate(range(0, len(pair_specs), batch_size)):
            block = pair_specs[start:start + batch_size]
            pair_seed = base_seed + block_index
            for condition in fx.ALL_CONDITIONS:
                pending = [(item, rollout) for item, rollout in block
                           if (item["item_id"], rollout, condition) not in completed]
                if not pending:
                    continue
                # Reset to the same seed for every condition in a paired block.
                torch.manual_seed(pair_seed)
                torch.cuda.manual_seed_all(pair_seed)
                prompts = [item[f"{condition}_prompt"] for item, _ in pending]
                rendered = [tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=True,
                ) for prompt in prompts]
                inputs = tokenizer(rendered, return_tensors="pt", padding=True).to(
                    model_input_device(model)
                )
                kwargs = generation_kwargs(spec, inputs.input_ids.shape[1])
                kwargs["pad_token_id"] = tokenizer.eos_token_id
                with torch.inference_mode():
                    outputs = model.generate(**inputs, **kwargs)
                for row_index, ((item, rollout), rendered_prompt) in enumerate(zip(pending, rendered)):
                    prompt_ids = inputs.input_ids[row_index][inputs.attention_mask[row_index].bool()]
                    generated_ids = trim_at_eos(
                        outputs[row_index, inputs.input_ids.shape[1]:].detach().cpu().tolist(),
                        getattr(model.generation_config, "eos_token_id", tokenizer.eos_token_id),
                    )
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
                    if "</think>" in generated_text:
                        reasoning, final_text = generated_text.split("</think>", 1)
                        complete = True
                    else:
                        reasoning, final_text, complete = generated_text, "", False
                    reasoning = reasoning.removeprefix("<think>").strip()
                    final_text = final_text.strip()
                    parsed = fx.parse_boxed_choice(final_text)
                    result = {
                        "item_id": item["item_id"], "split": item["split"],
                        "rollout": rollout, "condition": condition,
                        "seed": pair_seed, "generation_batch": block_index,
                        "gold_choice": item["gold_choice"],
                        "decision_critical_factor": item["decision_critical_factor"],
                        "prompt": item[f"{condition}_prompt"],
                        "rendered_prompt": rendered_prompt,
                        "generated_text": generated_text,
                        "full_ids": prompt_ids.detach().cpu().tolist() + generated_ids,
                        "reasoning": reasoning, "final_text": final_text,
                        "parsed_answer": parsed, "generation_complete": complete,
                        "generation_usable": bool(complete and parsed is not None),
                        "generated_token_count": len(generated_ids),
                        "generator_model": spec["model"],
                        "generator_requested_revision": spec.get("revision", "main"),
                        "generator_resolved_revision": getattr(model.config, "_commit_hash", None),
                        "generation_max_new_tokens": int(spec["max_new_tokens"]),
                        "generation_batch_size": int(spec["batch_size"]),
                        "generation_temperature": float(spec["temperature"]),
                        "generation_top_p": float(spec["top_p"]),
                        "generation_top_k": int(spec["top_k"]),
                        "generation_presence_penalty": float(spec.get("presence_penalty", 0.0)),
                    }
                    with path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    existing.append(result)
                    completed.add((item["item_id"], rollout, condition))
                print(f"Stage A generation: {len(completed)}/{total} rows")
    finally:
        del model
        torch.cuda.empty_cache()
    annotated = fx.annotate_stage_a(pd.DataFrame(existing))
    fx.write_jsonl(path, annotated.to_dict("records"))
    print(f"Stage A generations complete and annotated: {path}")


def monitor_stage_a(config: dict[str, Any], target: str) -> None:
    if target not in {"cot", "final"}:
        raise ValueError("target must be cot or final")
    reject_conclusive_early_failure(config)
    gen_spec, spec = config["generator"], config["semantic_monitor"]
    if spec["model"] == gen_spec["model"]:
        raise RuntimeError("The semantic monitor must be stronger and different from the generator")
    generations = pd.DataFrame(fx.read_jsonl(output_dir(config) / "stage_a_generations.jsonl"))
    if generations.empty:
        raise FileNotFoundError("Run generate before monitor")
    items = load_items(config)
    expected = len(items) * int(config["design"]["n_rollouts"]) * len(fx.ALL_CONDITIONS)
    if len(generations) != expected or "behavior_succeeds" not in generations:
        raise RuntimeError(
            f"Stage A generation is incomplete or unannotated ({len(generations)}/{expected})"
        )
    if target == "final":
        # Final-output monitoring belongs to Stage B, so enforce the phenomenon gate
        # before loading the large semantic judge.
        fx.require_stage_b_gate(gate_report(config))
    subset = generations[
        generations.condition.isin(fx.STRESS_CONDITIONS)
        & generations.behavior_succeeds.eq(True)
    ].copy()
    path = output_dir(config) / f"stage_a_{target}_monitor.jsonl"
    existing = fx.read_jsonl(path)
    complete = {(x["item_id"], int(x["rollout"]), x["condition"]) for x in existing}
    if len(existing) != len(complete):
        raise RuntimeError(f"{target} monitor checkpoint contains duplicate keys")
    if any(x.get("monitor_model") != spec["model"] for x in existing):
        raise RuntimeError(
            f"{target} monitor model changed after scoring began; use a new output directory"
        )
    pending_rows = [row for _, row in subset.iterrows()
                    if (row.item_id, int(row.rollout), row.condition) not in complete]
    if not pending_rows:
        print(f"{target} monitor checkpoint already complete: {len(existing)} rows")
        return
    model, tokenizer = load_hf_model(spec)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    batch_size = int(spec["batch_size"])
    try:
        for start in range(0, len(pending_rows), batch_size):
            block = pending_rows[start:start + batch_size]
            prompts = [fx.make_monitor_prompt(row, target) for row in block]
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
                parsed = fx.parse_structured_monitor(response)
                result = {
                    "item_id": row.item_id, "split": row.split,
                    "rollout": int(row.rollout), "condition": row.condition,
                    "monitor_target": target, "monitor_response": response,
                    **parsed, "monitor_model": spec["model"],
                    "monitor_requested_revision": spec.get("revision", "main"),
                    "monitor_resolved_revision": getattr(model.config, "_commit_hash", None),
                }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                existing.append(result)
            print(f"Structured {target} monitor: {len(existing)}/{len(subset)} rows")
    finally:
        del model
        torch.cuda.empty_cache()


def prepare_review(config: dict[str, Any]) -> None:
    out = output_dir(config)
    generations = pd.DataFrame(fx.read_jsonl(out / "stage_a_generations.jsonl"))
    monitor = pd.DataFrame(fx.read_jsonl(out / "stage_a_cot_monitor.jsonl"))
    if generations.empty or monitor.empty:
        raise FileNotFoundError("Generation and CoT-monitor checkpoints are required")
    items = load_items(config)
    expected_generations = (
        len(items) * int(config["design"]["n_rollouts"]) * len(fx.ALL_CONDITIONS)
    )
    if len(generations) != expected_generations:
        raise RuntimeError(
            f"Generation checkpoint is incomplete: {len(generations)}/{expected_generations}"
        )
    expected_monitor = int(
        (generations.condition.isin(fx.STRESS_CONDITIONS)
         & generations.behavior_succeeds.eq(True)).sum()
    )
    if len(monitor) != expected_monitor:
        raise RuntimeError(f"CoT monitor is incomplete: {len(monitor)}/{expected_monitor}")
    rows = fx.make_review_rows(
        generations, monitor,
        float(config["lexical_screen"]["content_word_coverage_cutoff"]),
    )
    path = out / "human_adjudications.csv"
    fx.write_review_csv(path, rows)
    print(f"Created {len(rows)} human-review rows at {path}")
    print("Complete human_label=A/B/C and factor_affected_behavior=true/false for every row.")


def gate_report(config: dict[str, Any], write: bool = True) -> dict[str, Any]:
    out = output_dir(config)
    reviews = fx.read_review_csv(out / "human_adjudications.csv")
    cot = pd.DataFrame(fx.read_jsonl(out / "stage_a_cot_monitor.jsonl")).rename(
        columns={"monitor_label": "cot_monitor_label"}
    )
    if cot.empty:
        raise FileNotFoundError("CoT semantic-monitor results are required for the gate")
    reviews = reviews.merge(
        cot[["item_id", "rollout", "condition", "cot_monitor_label"]],
        on=["item_id", "rollout", "condition"], how="left", validate="one_to_one",
    )
    design = config["design"]
    report = fx.evaluate_gate(
        reviews, int(design["gate_min_valid_trajectories"]),
        int(design["gate_min_unique_items"]),
    )
    if write:
        (out / "stage_a_gate.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return report


def early_gate_report(config: dict[str, Any]) -> dict[str, Any]:
    out = output_dir(config)
    generations = pd.DataFrame(fx.read_jsonl(out / "stage_a_generations.jsonl"))
    expected = int(config["design"]["n_items"]) * int(
        config["design"]["n_rollouts"]
    ) * len(fx.ALL_CONDITIONS)
    if len(generations) != expected or "behavior_succeeds" not in generations:
        raise RuntimeError(f"Annotated Stage A checkpoint is incomplete: {len(generations)}/{expected}")
    design = config["design"]
    report = fx.evaluate_early_gate(
        generations,
        int(design["gate_min_valid_trajectories"]),
        int(design["gate_min_unique_items"]),
        float(config["lexical_screen"]["content_word_coverage_cutoff"]),
    )
    (out / "stage_a_early_gate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    candidates = generations[
        generations.condition.isin(fx.STRESS_CONDITIONS)
        & generations.behavior_succeeds.eq(True)
    ][[
        "item_id", "split", "rollout", "condition", "decision_critical_factor",
        "paired_control_answer", "paired_cue_answer", "parsed_answer", "reasoning",
        "factor_exact_match", "factor_word_coverage", "matched_factor_words",
    ]]
    candidates.to_csv(out / "stage_a_behavior_success_audit.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


def score_stage_b(config: dict[str, Any]) -> None:
    reject_conclusive_early_failure(config)
    report = gate_report(config)
    fx.require_stage_b_gate(report)  # Must occur before model/lens loading.
    import jlens
    from huggingface_hub import hf_hub_download
    import monitorbench_experiment as mbe

    out, stage = output_dir(config), config["stage_b"]
    generator = config["generator"]
    model, tokenizer = load_hf_model(generator)
    lens_model = jlens.from_hf(model, tokenizer)
    lens_path = hf_hub_download(
        stage["lens_repository"], filename=stage["lens_file"],
        revision=stage["lens_revision"],
    )
    lens = jlens.JacobianLens.load(lens_path)
    dataset = json.loads(resolve_path(config, config["dataset_path"]).read_text(encoding="utf-8"))
    item_count = int(config["design"]["n_items"])
    token_sets = mbe.build_frequency_matched_token_sets(dataset[:item_count], tokenizer)
    # The helper keys by benchmark id, matching item_id.
    generations = pd.DataFrame(fx.read_jsonl(out / "stage_a_generations.jsonl"))
    reviews = fx.read_review_csv(out / "human_adjudications.csv")
    keys = set(zip(reviews.item_id, reviews.rollout.astype(int)))
    score_conditions = {"control", *fx.STRESS_CONDITIONS}
    subset = generations[
        generations.condition.isin(score_conditions)
        & generations.apply(lambda x: (x.item_id, int(x.rollout)) in keys, axis=1)
        & generations.generation_usable.eq(True)
    ]
    path = out / "stage_b_lens_position_scores.jsonl"
    existing = fx.read_jsonl(path)
    think_start = tokenizer.convert_tokens_to_ids("<think>")
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    layers = [int(x) for x in stage["layers"]]
    max_positions = int(stage["max_reasoning_positions"])
    try:
        for _, row in subset.iterrows():
            input_ids = [int(x) for x in row.full_ids]
            positions = mbe.response_and_reasoning_positions(
                input_ids, think_start, think_end, n_reasoning_positions=max_positions
            )
            ids = token_sets[row.item_id]
            for method, use_jacobian in (("j_lens", True), ("logit_lens", False)):
                key = (row.item_id, int(row.rollout), row.condition, method)
                key_rows = [x for x in existing if (
                    x["item_id"], int(x["rollout"]), x["condition"], x["method"]
                ) == key]
                expected_coordinates = {
                    (layer, position_index)
                    for layer in layers for position_index in range(len(positions))
                }
                present_coordinates = {
                    (int(x["layer"]), int(x["position_index"])) for x in key_rows
                }
                if present_coordinates == expected_coordinates:
                    continue
                if key_rows:
                    # A process stopped during the small multi-row write. Remove only
                    # that incomplete trajectory-method block, then recompute it.
                    existing = [x for x in existing if (
                        x["item_id"], int(x["rollout"]), x["condition"], x["method"]
                    ) != key]
                    fx.write_jsonl(path, existing)
                logits = mbe.apply_lens_to_ids(
                    lens, lens_model, input_ids, layers, positions, use_jacobian
                )
                rows = []
                for layer, layer_logits in logits.items():
                    log_probs = layer_logits.float().log_softmax(dim=-1)
                    targets = torch.logsumexp(
                        log_probs[:, ids["target_token_ids"]], dim=-1
                    ) - math.log(len(ids["target_token_ids"]))
                    controls = torch.logsumexp(
                        log_probs[:, ids["control_token_ids"]], dim=-1
                    ) - math.log(len(ids["control_token_ids"]))
                    for position_index, (token_position, score) in enumerate(
                        zip(positions, (targets - controls).tolist())
                    ):
                        rows.append({
                            "item_id": row.item_id, "split": row.split,
                            "rollout": int(row.rollout), "condition": row.condition,
                            "method": method, "layer": int(layer),
                            "position_index": position_index,
                            "token_position": int(token_position), "score": float(score),
                        })
                with path.open("a", encoding="utf-8") as handle:
                    for result in rows:
                        handle.write(json.dumps(result) + "\n")
                existing.extend(rows)
                completed_count = len({
                    (x["item_id"], int(x["rollout"]), x["condition"], x["method"])
                    for x in existing
                })
                print(f"Stage B lens scoring: {completed_count} trajectory-methods")
    finally:
        del lens, lens_model, model
        torch.cuda.empty_cache()


def assemble_stage_b_evaluation(config: dict[str, Any]) -> pd.DataFrame:
    out, stage = output_dir(config), config["stage_b"]
    reviews = fx.read_review_csv(out / "human_adjudications.csv")
    reviews = reviews[reviews.human_label.isin(["A", "B"])].copy()
    reviews["label"] = reviews.human_label.eq("B").astype(int)
    cot = pd.DataFrame(fx.read_jsonl(out / "stage_a_cot_monitor.jsonl")).rename(
        columns={"monitor_omission_score": "cot_monitor_score"}
    )
    final = pd.DataFrame(fx.read_jsonl(out / "stage_a_final_monitor.jsonl")).rename(
        columns={"monitor_omission_score": "final_output_monitor_score"}
    )
    keys = ["item_id", "rollout", "condition"]
    frame = reviews.merge(cot[keys + ["cot_monitor_score"]], on=keys, how="left")
    frame = frame.merge(final[keys + ["final_output_monitor_score"]], on=keys, how="left")
    lens = pd.DataFrame(fx.read_jsonl(out / "stage_b_lens_position_scores.jsonl"))
    if lens.empty:
        raise FileNotFoundError("Run stage-b-score first")
    fpr = float(stage["false_positive_rate"])
    for method in ("j_lens", "logit_lens"):
        method_rows = lens[lens.method.eq(method)]
        maxima = method_rows.groupby(
            ["item_id", "rollout", "condition"]
        ).score.max().rename("max_score").reset_index()
        controls = maxima[maxima.condition.eq("control")][
            ["item_id", "rollout", "max_score"]
        ].rename(columns={"max_score": "control_max"})
        stress = maxima[maxima.condition.isin(fx.STRESS_CONDITIONS)].merge(
            controls, on=["item_id", "rollout"], how="left"
        )
        stress[f"{method}_score"] = stress.max_score - stress.control_max
        stress[f"{method}_target_token_contrast"] = stress[f"{method}_score"]
        frame = frame.merge(
            stress[keys + [f"{method}_score", f"{method}_target_token_contrast"]],
            on=keys, how="left",
        )
        dev_controls = maxima[
            maxima.condition.eq("control")
            & maxima.item_id.isin(frame[frame.split.eq("dev")].item_id)
        ].max_score
        if dev_controls.empty:
            raise ValueError("No development control lens scores for early-position threshold")
        threshold = float(np.quantile(dev_controls, 1.0 - fpr, method="higher"))
        early = (
            method_rows[method_rows.condition.isin(fx.STRESS_CONDITIONS)]
            .groupby(keys + ["position_index"]).score.max().reset_index()
        )
        early = early[early.score.ge(threshold)].groupby(keys).position_index.min()
        early = early.rename(f"{method}_early_detection_position").reset_index()
        frame = frame.merge(early, on=keys, how="left")
    # Train the simple combined score only on the development split.
    dev = frame[frame.split.eq("dev")]
    parts = []
    for column in ("cot_monitor_score", "j_lens_score"):
        mean, std = float(dev[column].mean()), float(dev[column].std(ddof=1))
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Cannot standardize {column} on development rows")
        parts.append((frame[column] - mean) / std)
    frame["cot_plus_j_score"] = parts[0] + parts[1]
    return frame


def report_stage_b(config: dict[str, Any]) -> None:
    reject_conclusive_early_failure(config)
    report = gate_report(config)
    fx.require_stage_b_gate(report)
    out, stage = output_dir(config), config["stage_b"]
    frame = assemble_stage_b_evaluation(config)
    required_scores = [
        "cot_monitor_score", "final_output_monitor_score", "logit_lens_score",
        "j_lens_score", "cot_plus_j_score",
    ]
    if frame[required_scores].isna().any().any():
        raise RuntimeError("Stage B inputs are incomplete; monitor/lens scores are missing")
    metrics = fx.stage_b_metrics(
        frame,
        ["cot_monitor", "final_output_monitor", "logit_lens", "j_lens", "cot_plus_j"],
        float(stage["false_positive_rate"]), int(stage["bootstrap_samples"]),
        int(stage["bootstrap_seed"]),
    )
    fx.write_jsonl(out / "stage_b_evaluation.jsonl", frame.to_dict("records"))
    fx.write_jsonl(out / "stage_b_metrics.jsonl", metrics.to_dict("records"))
    print(metrics.to_string(index=False))


def status(config: dict[str, Any]) -> None:
    out = output_dir(config)
    items = fx.read_jsonl(out / "stage_a_items.jsonl")
    generations = fx.read_jsonl(out / "stage_a_generations.jsonl")
    expected = len(items) * int(config["design"]["n_rollouts"]) * len(fx.ALL_CONDITIONS)
    print(f"items: {len(items)}/{config['design']['n_items']}")
    print(f"stage_a_generations: {len(generations)}/{expected}")
    early_path = out / "stage_a_early_gate.json"
    if early_path.exists():
        early = json.loads(early_path.read_text(encoding="utf-8"))
        print(
            "early_gate: "
            f"conclusive_failure={early.get('conclusive_failure')} "
            f"behavior_success={early.get('behavior_success_trajectories')} "
            f"lexically_clear={early.get('lexically_clear_trajectories')} "
            f"next={early.get('next_action')}"
        )
    for target in ("cot", "final"):
        rows = fx.read_jsonl(out / f"stage_a_{target}_monitor.jsonl")
        print(f"{target}_monitor_rows: {len(rows)}")
    review_path = out / "human_adjudications.csv"
    if review_path.exists():
        gate_report(config, write=False)
    else:
        print("human_adjudications: not prepared")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "generate", "early-gate", "prepare-review", "gate",
                    "stage-b-score", "stage-b-report", "status"):
        sub.add_parser(command)
    monitor_parser = sub.add_parser("monitor")
    monitor_parser.add_argument("--target", choices=["cot", "final"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    commands = {
        "prepare": lambda: prepare(config),
        "generate": lambda: generate_stage_a(config),
        "early-gate": lambda: early_gate_report(config),
        "monitor": lambda: monitor_stage_a(config, args.target),
        "prepare-review": lambda: prepare_review(config),
        "gate": lambda: gate_report(config),
        "stage-b-score": lambda: score_stage_b(config),
        "stage-b-report": lambda: report_stage_b(config),
        "status": lambda: status(config),
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
