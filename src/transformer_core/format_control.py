"""Source-aligned prose control with fixed byte-length, per-target histories."""

import hashlib
import html
import json
import math
from html.parser import HTMLParser
from pathlib import Path

import torch
from torch.nn import functional as F

from .language_model import fit_bigram
from .stability import configure_cpu, file_digest, summary, write_json
from .transfer import checked_models, verified_bytes

DROP = frozenset({"tcl", "script", "style", "pre", "code"})
BLOCK = frozenset(
    {
        "p",
        "br",
        "div",
        "table",
        "tr",
        "td",
        "th",
        "ul",
        "ol",
        "li",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "blockquote",
        "hr",
        "title",
    }
)
CLOSE_PARAGRAPH = BLOCK - {"p", "br", "td", "th"}
ASCII_SPACE = " \t\r\n\f"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ProseProjection(HTMLParser):
    """Conservative projection, not SQLite's generator or a browser renderer."""

    def __init__(self, raw):
        super().__init__(convert_charrefs=False)
        self.raw = raw
        self.text = raw.decode("utf-8", errors="strict")
        self.char_bytes = [0]
        for char in self.text:
            self.char_bytes.append(self.char_bytes[-1] + len(char.encode("utf-8")))
        self.lines = [0] + [i + 1 for i, char in enumerate(self.text) if char == "\n"]
        self.output, self.origins, self.targets = bytearray(), [], []
        self.suppressed = []
        self.paragraph = False

    def char_position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def origin(self):
        return self.char_bytes[self.char_position()]

    def space(self, origin):
        if self.output and self.output[-1] != 32:
            self.output.append(32)
            self.origins.append(origin)
            self.targets.append(False)

    def emit(self, char, origin, literal=False):
        if char in ASCII_SPACE:
            self.space(origin)
            return
        data = char.encode("utf-8")
        is_target = literal and self.paragraph and char.isascii() and char.isalpha()
        for j, byte in enumerate(data):
            self.output.append(byte)
            self.origins.append(origin + j if literal else origin)
            self.targets.append(is_target)

    def handle_starttag(self, tag, attrs):
        if self.suppressed:
            if tag in DROP:
                self.suppressed.append(tag)
            return
        if tag in DROP:
            self.space(self.origin())
            self.suppressed.append(tag)
            if tag == "pre":
                self.paragraph = False
            return
        if tag in BLOCK:
            self.space(self.origin())
        if tag == "p":
            self.paragraph = True
        elif tag in CLOSE_PARAGRAPH:
            self.paragraph = False

    def handle_endtag(self, tag):
        if self.suppressed:
            if tag == self.suppressed[-1]:
                self.suppressed.pop()
                if not self.suppressed:
                    self.space(self.origin())
            return
        if tag in BLOCK:
            self.space(self.origin())
        if tag == "p" or tag in CLOSE_PARAGRAPH:
            self.paragraph = False

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data):
        if self.suppressed:
            return
        start = self.char_position()
        if self.text[start : start + len(data)] != data:
            raise ValueError("parser text/source alignment mismatch")
        for j, char in enumerate(data):
            self.emit(char, self.char_bytes[start + j], literal=True)

    def entity(self, text):
        if not self.suppressed:
            for char in html.unescape(text):
                self.emit(char, self.origin(), literal=False)

    def handle_entityref(self, name):
        self.entity("&" + name + ";")

    def handle_charref(self, name):
        self.entity("&#" + name + ";")

    def render(self):
        self.feed(self.text)
        self.close()
        if self.suppressed:
            raise ValueError("unclosed suppressed source span")
        if self.output and self.output[-1] == 32:
            self.output.pop()
            self.origins.pop()
            self.targets.pop()
        return bytes(self.output), self.origins, self.targets


def spaced_indexes(size, cap):
    count = min(size, cap)
    if count < 1:
        return []
    if count == 1:
        return [0]
    return [i * (size - 1) // (count - 1) for i in range(count)]


def document_pairs(raw, item, config):
    projected, origins, targets = ProseProjection(raw).render()
    width = config["context_bytes"]
    eligible = {"changed": [], "identical": []}
    counts = dict(
        literal_paragraph_ascii_targets=sum(targets),
        early_raw=0,
        early_projected=0,
        early_either=0,
    )
    for j, target in enumerate(targets):
        if not target:
            continue
        i = origins[j]
        if raw[i] != projected[j]:
            raise ValueError("target byte not source aligned")
        if i < width or j < width:
            counts["early_raw"] += i < width
            counts["early_projected"] += j < width
            counts["early_either"] += 1
            continue
        if any(origin >= i for origin in origins[j - width : j]):
            raise ValueError("projected history contains current/future source bytes")
        a, b = raw[i - width : i], projected[j - width : j]
        kind = "identical" if a == b else "changed"
        eligible[kind].append(
            dict(
                raw_offset=i,
                projected_offset=j,
                target=raw[i],
                raw_history_sha256=digest(a),
                projected_history_sha256=digest(b),
                same_predecessor=a[-1] == b[-1],
            )
        )
    groups = {}
    for kind in eligible:
        counts[kind + "_eligible"] = len(eligible[kind])
        indexes = spaced_indexes(len(eligible[kind]), config[kind + "_cap_per_document"])
        groups[kind] = [eligible[kind][i] for i in indexes]
        counts[kind + "_sampled"] = len(indexes)
        counts[kind + "_same_predecessor"] = sum(p["same_predecessor"] for p in groups[kind])
    return dict(
        id=item["id"],
        raw_sha256=digest(raw),
        projected_sha256=digest(projected),
        raw_bytes=len(raw),
        projected_bytes=len(projected),
        counts=counts,
        pairs=groups,
    )


def prepare_pairs(corpus_manifest, corpus_sha, data_dir, config, provenance):
    if file_digest(corpus_manifest) != corpus_sha:
        raise ValueError("pinned SQLite manifest changed")
    source = json.loads(Path(corpus_manifest).read_text())
    docs = [
        document_pairs(verified_bytes(data_dir, item), item, config)
        for item in source["documents"]
    ]
    if any(not d["pairs"]["changed"] or not d["pairs"]["identical"] for d in docs):
        raise ValueError("each document must provide both frozen strata")
    return dict(
        schema_version=1,
        context_bytes=config["context_bytes"],
        corpus_manifest_sha256=corpus_sha,
        provenance=provenance,
        documents=docs,
    )


def histories(raw, projected, pairs, width):
    a, b, y = [], [], []
    for pair in pairs:
        i, j = pair["raw_offset"], pair["projected_offset"]
        left, right = raw[max(0, i - width) : i], projected[max(0, j - width) : j]
        if len(left) != width or len(right) != width:
            raise ValueError("history length differs from fixed byte budget")
        if raw[i] != projected[j] or raw[i] != pair["target"]:
            raise ValueError("paired target changed")
        if (
            digest(left) != pair["raw_history_sha256"]
            or digest(right) != pair["projected_history_sha256"]
        ):
            raise ValueError("paired history digest changed")
        a.append(list(left))
        b.append(list(right))
        y.append(pair["target"])
    return torch.tensor(a, dtype=torch.long), torch.tensor(b, dtype=torch.long), torch.tensor(y)


@torch.no_grad()
def last_target_losses(model, inputs, targets, batch_size):
    model.eval()
    values = []
    for start in range(0, len(targets), batch_size):
        batch = inputs[start : start + batch_size]
        logits = model(batch)[:, -1, :].double()
        losses = F.cross_entropy(logits, targets[start : start + batch_size], reduction="none")
        if not torch.isfinite(losses).all():
            raise ValueError("nonfinite format-control loss")
        values.extend(losses.tolist())
    return values


def metric(values):
    total = math.fsum(values)
    return dict(
        count=len(values),
        nll_sum=total,
        nll=total / len(values),
        byte_perplexity=math.exp(total / len(values)),
    )


def condition_metrics(left, right):
    if not left or len(left) != len(right):
        raise ValueError("paired losses must be nonempty and aligned")
    return dict(
        raw=metric(left),
        projected=metric(right),
        projected_minus_raw_nll=math.fsum(b - a for a, b in zip(left, right)) / len(left),
        max_abs_pair_difference=max(abs(b - a) for a, b in zip(left, right)),
    )


def aggregate_format(runs, seeds):
    indexed = {(r["seed"], r["arm"]): r for r in runs}
    if len(indexed) != len(runs) or set(indexed) != {
        (s, a) for s in seeds for a in ("sinusoidal", "no_position")
    }:
        raise ValueError("missing/duplicate format seed-arm roster")
    result = {}
    for kind in ("changed", "identical"):
        arms = {}
        for arm in ("sinusoidal", "no_position"):
            rows = [indexed[s, arm]["pooled"][kind]["model"] for s in seeds]
            arms[arm] = dict(
                raw_nll=summary([r["raw"]["nll"] for r in rows]),
                projected_nll=summary([r["projected"]["nll"] for r in rows]),
                projection_effect=summary([r["projected_minus_raw_nll"] for r in rows]),
                raw_byte_perplexity=summary([r["raw"]["byte_perplexity"] for r in rows]),
                projected_byte_perplexity=summary(
                    [r["projected"]["byte_perplexity"] for r in rows]
                ),
            )
        pairs = [
            dict(
                seed=s,
                no_position_minus_sinusoidal_projection_effect=(
                    indexed[s, "no_position"]["pooled"][kind]["model"][
                        "projected_minus_raw_nll"
                    ]
                    - indexed[s, "sinusoidal"]["pooled"][kind]["model"][
                        "projected_minus_raw_nll"
                    ]
                ),
            )
            for s in seeds
        ]
        result[kind] = dict(
            arms=arms,
            paired_seeds=pairs,
            paired_effect_difference=summary(
                [p["no_position_minus_sinusoidal_projection_effect"] for p in pairs]
            ),
        )
    return result


def score_format(
    config,
    pair_manifest,
    selection_path,
    checkpoint_dir,
    source_root,
    train,
    corpus_manifest,
    data_dir,
    output_path,
    provenance,
    checkpoint_origin="unspecified",
):
    if Path(output_path).exists():
        raise ValueError("format report already exists")
    regenerated = prepare_pairs(
        corpus_manifest, config["corpus_manifest_sha256"], data_dir, config, provenance
    )
    if regenerated != pair_manifest:
        raise ValueError("frozen pair manifest does not match regenerated data/source")
    configure_cpu()
    selection, models, identity = checked_models(
        selection_path, config["original_selection_sha256"], checkpoint_dir, source_root
    )
    if config["context_bytes"] != selection["config"]["context"]:
        raise ValueError("format history budget differs from frozen model context")
    baseline = fit_bigram(train, alpha=selection["config"]["bigram_alpha"])
    source = json.loads(Path(corpus_manifest).read_text())
    prepared = []
    for item, doc in zip(source["documents"], pair_manifest["documents"]):
        raw = verified_bytes(data_dir, item)
        projected, _, _ = ProseProjection(raw).render()
        prepared.append(
            {
                kind: histories(raw, projected, doc["pairs"][kind], config["context_bytes"])
                for kind in ("changed", "identical")
            }
        )
    runs = []
    # Pair regeneration, every identity and all input histories pass before any forward.
    for row, model in zip(selection["runs"], models):
        documents = []
        pooled = {
            kind: {name: [[], []] for name in ("model", "bigram")}
            for kind in ("changed", "identical")
        }
        for doc, data in zip(pair_manifest["documents"], prepared):
            metrics = {}
            for kind, (left, right, target) in data.items():
                losses = {
                    "model": [
                        last_target_losses(model, x, target, config["batch_size"])
                        for x in (left, right)
                    ],
                    "bigram": [(-baseline[x[:, -1], target]).tolist() for x in (left, right)],
                }
                metrics[kind] = {}
                for name, (a, b) in losses.items():
                    m = condition_metrics(a, b)
                    if kind == "identical" and m["max_abs_pair_difference"] > 1e-12:
                        raise ValueError("identical-history control failed")
                    metrics[kind][name] = m
                    pooled[kind][name][0].extend(a)
                    pooled[kind][name][1].extend(b)
            documents.append(dict(id=doc["id"], strata=metrics))
        runs.append(
            dict(
                seed=row["seed"],
                arm=row["arm"],
                selected_step=row["selected_step"],
                documents=documents,
                pooled={
                    kind: {name: condition_metrics(*values) for name, values in names.items()}
                    for kind, names in pooled.items()
                },
            )
        )
    if file_digest(selection_path) != config["original_selection_sha256"]:
        raise ValueError("original selection changed during scoring")
    result = dict(
        schema_version=1,
        provenance=provenance,
        pair_manifest_canonical_sha256=digest(
            (json.dumps(pair_manifest, indent=2, allow_nan=False) + "\n").encode()
        ),
        checkpoint_origin=checkpoint_origin,
        checkpoint_identity=identity,
        context_bytes=config["context_bytes"],
        runs=runs,
        aggregate=aggregate_format(runs, selection["config"]["seeds"]),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result
