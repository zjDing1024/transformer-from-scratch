"""Original paired experiment with a separate, hash-locked scoring phase."""

import copy
import hashlib
import json
import math
import platform
import statistics
import struct
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .language_model import ByteLanguageModel, evaluate, fit_bigram, sample_batch


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    """Stable CPU float tensor digest without NumPy or pickle serialization."""
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        values = tensor.detach().cpu().flatten().tolist()
        h.update(struct.pack(f"<{len(values)}f", *values))
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_config(config):
    if len(set(config["seeds"])) != len(config["seeds"]) or len(config["seeds"]) < 2:
        raise ValueError("at least two distinct seeds required")
    if config["arms"] != {"sinusoidal": True, "no_position": False}:
        raise ValueError("exactly the paired position arms are required")
    for key in ("steps", "validate_every", "batch", "context"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["steps"] % config["validate_every"]:
        raise ValueError("final step must be a validation checkpoint")
    for key in ("learning_rate", "clip_norm", "bigram_alpha"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")


def configure_cpu():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train_candidate(train, validation, config, seed, arm, output_dir):
    """Only train/validation documents can enter this function, never test."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ByteLanguageModel(**config["model"], use_positions=config["arms"][arm])
    initial_sha = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    schedule = hashlib.sha256()
    best, selected, state, curve = float("inf"), None, None, []
    maximum_norm, clipped = 0.0, 0
    start = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        model.train()
        x, y = sample_batch(train, config["context"], config["batch"], generator)
        schedule.update(bytes(x.flatten().tolist()))
        schedule.update(bytes(y.flatten().tolist()))
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).reshape(-1, 256), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss in {seed}/{arm}/{step}; abort study")
        loss.backward()
        norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["clip_norm"], error_if_nonfinite=True
            )
        )
        maximum_norm = max(maximum_norm, norm)
        clipped += norm > config["clip_norm"]
        optimizer.step()
        if step % config["validate_every"] == 0:
            score = evaluate(model, validation, config["context"])["model"]
            if not math.isfinite(score["nll"]):
                raise RuntimeError(
                    f"nonfinite validation loss in {seed}/{arm}/{step}; abort study"
                )
            curve.append(dict(step=step, train_last_batch_nll=loss.item(), validation=score))
            if score["nll"] < best:
                best, selected = score["nll"], step
                state = copy.deepcopy(model.state_dict())
    if state is None:
        raise RuntimeError("no finite selected checkpoint")
    path = Path(output_dir) / f"{seed}-{arm}.pt"
    torch.save(dict(model=state, config=model.config, selected_step=selected), path)
    return dict(
        seed=seed,
        arm=arm,
        selected_step=selected,
        validation_nll=best,
        steps=config["steps"],
        target_exposures=config["steps"] * config["batch"] * config["context"],
        parameter_count=sum(p.numel() for p in model.parameters()),
        initial_state_sha256=initial_sha,
        sample_schedule_sha256=schedule.hexdigest(),
        selected_state_sha256=state_digest(state),
        checkpoint=path.name,
        checkpoint_sha256=file_digest(path),
        max_preclip_gradient_norm=maximum_norm,
        clipped_steps=clipped,
        finite_training_steps=config["steps"],
        curve=curve,
        training_and_validation_seconds=time.perf_counter() - start,
    )


def train_study(train, validation, config, output_dir, provenance):
    validate_config(config)
    configure_cpu()
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; retain prior attempts")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "started.json", dict(config=config, provenance=provenance))
    runs = []
    for seed in config["seeds"]:
        pair = []
        for arm in config["arms"]:
            row = train_candidate(train, validation, config, seed, arm, output_dir)
            runs.append(row)
            pair.append(row)
            write_json(output_dir / "training-progress.json", runs)
        for field in ("initial_state_sha256", "sample_schedule_sha256", "parameter_count"):
            if pair[0][field] != pair[1][field]:
                raise RuntimeError(f"paired control failed: {field}")
    selection = dict(
        schema_version=1,
        config=config,
        provenance=provenance,
        runs=runs,
        environment=dict(
            python=platform.python_version(), torch=torch.__version__, device="cpu", threads=1
        ),
        status="all candidates selected using validation only; external scoring not started",
    )
    lock = output_dir / "selection.json"
    write_json(lock, selection)
    return file_digest(lock)


def summary(values):
    return dict(
        mean=statistics.mean(values),
        sample_sd=statistics.stdev(values),
        minimum=min(values),
        maximum=max(values),
        n=len(values),
    )


def aggregate_runs(runs, seeds):
    expected = {(seed, arm) for seed in seeds for arm in ("sinusoidal", "no_position")}
    keys = [(r["seed"], r["arm"]) for r in runs]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("missing/duplicate/unexpected seed-arm pair")
    arms = {
        arm: {
            metric: summary([r["pooled"]["model"][metric] for r in runs if r["arm"] == arm])
            for metric in ("nll", "perplexity", "bits_per_byte")
        }
        for arm in ("sinusoidal", "no_position")
    }
    indexed = {(r["seed"], r["arm"]): r for r in runs}
    pairs = [
        dict(
            seed=s,
            no_position_minus_sinusoidal_nll=(
                indexed[s, "no_position"]["pooled"]["model"]["nll"]
                - indexed[s, "sinusoidal"]["pooled"]["model"]["nll"]
            ),
        )
        for s in seeds
    ]
    return dict(
        arms=arms,
        paired_differences=pairs,
        paired_nll_summary=summary([p["no_position_minus_sinusoidal_nll"] for p in pairs]),
    )


def pool_documents(documents):
    """Pool sums/counts, not an unweighted average of document means."""
    pooled = {}
    for name in ("model", "bigram"):
        count = sum(d["metrics"][name]["tokens"] for d in documents)
        loss = sum(d["metrics"][name]["nll_sum"] for d in documents)
        mean = loss / count
        pooled[name] = dict(
            tokens=count,
            nll_sum=loss,
            nll=mean,
            perplexity=math.exp(mean),
            bits_per_byte=mean / math.log(2),
        )
    return pooled


def score_study(train, heldout, document_ids, config, output_dir, provenance, selection_sha):
    """Reject any changed selection/checkpoint before first held-out model score."""
    validate_config(config)
    configure_cpu()
    output_dir = Path(output_dir)
    if (output_dir / "report.json").exists():
        raise ValueError("scoring report already exists; do not overwrite held-out evidence")
    if not heldout or len(heldout) != len(document_ids):
        raise ValueError("held-out IDs and documents must be nonempty and aligned")
    lock = output_dir / "selection.json"
    if file_digest(lock) != selection_sha:
        raise ValueError("selection digest mismatch")
    selection = json.loads(lock.read_text())
    if selection["config"] != config or selection["provenance"] != provenance:
        raise ValueError("config/source/manifest provenance changed after selection")
    expected = {(s, a) for s in config["seeds"] for a in config["arms"]}
    keys = [(r["seed"], r["arm"]) for r in selection["runs"]]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete or duplicate selected roster")
    models = []
    for row in selection["runs"]:
        name = f"{row['seed']}-{row['arm']}.pt"
        if (
            row["checkpoint"] != name
            or file_digest(output_dir / name) != row["checkpoint_sha256"]
        ):
            raise ValueError("checkpoint digest/name mismatch")
        checkpoint = torch.load(output_dir / name, map_location="cpu", weights_only=True)
        expected_model = dict(config["model"], use_positions=config["arms"][row["arm"]])
        if (
            checkpoint["config"] != expected_model
            or checkpoint["selected_step"] != row["selected_step"]
        ):
            raise ValueError("checkpoint metadata mismatch")
        if state_digest(checkpoint["model"]) != row["selected_state_sha256"]:
            raise ValueError("selected state digest mismatch")
        model = ByteLanguageModel(**expected_model)
        model.load_state_dict(checkpoint["model"])
        models.append(model)
    # No held-out model scoring has occurred until every candidate passes checks.
    baseline = fit_bigram(train, alpha=config["bigram_alpha"])
    runs = []
    for row, model in zip(selection["runs"], models):
        documents = [
            dict(id=name, metrics=evaluate(model, [doc], config["context"], baseline))
            for name, doc in zip(document_ids, heldout)
        ]
        runs.append(
            dict(
                seed=row["seed"],
                arm=row["arm"],
                selected_step=row["selected_step"],
                documents=documents,
                pooled=pool_documents(documents),
            )
        )
    result = dict(
        schema_version=1,
        selection_sha256=selection_sha,
        config=config,
        provenance=provenance,
        environment=selection["environment"],
        runs=runs,
        aggregate=aggregate_runs(runs, config["seeds"]),
    )
    write_json(output_dir / "report.json", result)
    return result
