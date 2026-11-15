import json

import pytest
import torch

from transformer_core import format_control as fc
from transformer_core.language_model import ByteLanguageModel
from transformer_core.stability import file_digest, state_digest, write_json


def test_projection_literal_mapping_entities_and_suppression():
    raw = (
        "<title>标题</title><p>A <b>B</b>&amp; C &#65; "
        "<code>DROP</code>D</p><tcl>exec BAD</tcl><p>éE</p>"
    ).encode()
    data, origins, eligible = fc.ProseProjection(raw).render()
    assert data.decode() == "标题 A B& C A D éE"
    actual = [(data[j], origins[j]) for j in range(len(data)) if eligible[j]]
    assert bytes(x for x, _ in actual) == b"ABCDE"
    assert all(raw[offset] == value for value, offset in actual)
    assert actual[-1][1] == raw.index(b"E</p>")
    assert len(data) == len(origins) == len(eligible)
    assert all(a <= b for a, b in zip(origins, origins[1:]))


def test_unclosed_suppressed_span_fails():
    with pytest.raises(ValueError, match="unclosed"):
        fc.ProseProjection(b"<p>abc</p><tcl>bad").render()


@pytest.mark.parametrize(
    "size,cap,expected", [(0, 3, []), (1, 8, [0]), (10, 3, [0, 4, 9]), (4, 8, [0, 1, 2, 3])]
)
def test_equal_spacing(size, cap, expected):
    assert fc.spaced_indexes(size, cap) == expected


def test_pairs_have_fixed_history_and_no_current_or_future_occurrence():
    raw = b"<p>" + b"Abcde fghij " * 15 + b"<b>K</b> lmno " * 4 + b"</p>"
    config = dict(context_bytes=16, changed_cap_per_document=10, identical_cap_per_document=7)
    row = fc.document_pairs(raw, dict(id="test"), config)
    rendered, origins, _ = fc.ProseProjection(raw).render()
    assert row["counts"]["early_either"] > 0
    for kind, pairs in row["pairs"].items():
        a, b, y = fc.histories(raw, rendered, pairs, 16)
        assert a.shape == b.shape == (len(pairs), 16)
        for p, target, left, right in zip(pairs, y, a, b):
            assert raw[p["raw_offset"]] == int(target) == rendered[p["projected_offset"]]
            assert (
                max(origins[p["projected_offset"] - 16 : p["projected_offset"]])
                < p["raw_offset"]
            )
            assert torch.equal(left, right) == (kind == "identical")
            # Their offsets modulo the old evaluation block need not agree.
    assert row["counts"]["changed_sampled"] == 10
    assert row["counts"]["identical_sampled"] == 7


def test_last_target_is_scored_without_scoring_other_input_bytes():
    class Uniform(torch.nn.Module):
        def forward(self, inputs):
            assert inputs.shape[1] == 64
            return torch.zeros((*inputs.shape, 256))

    x = torch.ones((7, 64), dtype=torch.long)
    y = torch.arange(7)
    a = fc.last_target_losses(Uniform(), x, y, 3)
    assert len(a) == 7
    assert a == pytest.approx([torch.log(torch.tensor(256.0, dtype=torch.float64)).item()] * 7)
    with pytest.raises(ValueError, match="aligned"):
        fc.condition_metrics([1.0], [1.0, 2.0])


@pytest.fixture
def tiny_control(tmp_path):
    raw = (
        b"<p>"
        + b"A simple paragraph with letters and spaces. " * 5
        + b"<b>More</b> prose. " * 5
        + b"</p>"
    )
    item = dict(id="one", upstream_path="one.in", size_bytes=len(raw), sha256=fc.digest(raw))
    (tmp_path / "one.in").write_bytes(raw)
    manifest = tmp_path / "corpus.json"
    write_json(manifest, dict(documents=[item]))
    original = dict(
        seeds=[1, 2],
        arms=dict(sinusoidal=True, no_position=False),
        model=dict(width=4, heads=1, hidden=8, layers=1),
        context=8,
        bigram_alpha=0.1,
    )
    rows = []
    for seed in original["seeds"]:
        for arm, use_positions in original["arms"].items():
            torch.manual_seed(seed)
            model = ByteLanguageModel(**original["model"], use_positions=use_positions)
            name = f"{seed}-{arm}.pt"
            torch.save(
                dict(config=model.config, model=model.state_dict(), selected_step=1),
                tmp_path / name,
            )
            rows.append(
                dict(
                    seed=seed,
                    arm=arm,
                    checkpoint=name,
                    selected_step=1,
                    checkpoint_sha256=file_digest(tmp_path / name),
                    selected_state_sha256=state_digest(model.state_dict()),
                )
            )
    selection = tmp_path / "selection.json"
    write_json(selection, dict(config=original, runs=rows, provenance=dict(sources={})))
    config = dict(
        context_bytes=8,
        batch_size=3,
        changed_cap_per_document=5,
        identical_cap_per_document=4,
        original_selection_sha256=file_digest(selection),
        corpus_manifest_sha256=file_digest(manifest),
    )
    provenance = dict(synthetic=True)
    pairs = fc.prepare_pairs(manifest, file_digest(manifest), tmp_path, config, provenance)
    return tmp_path, manifest, selection, config, provenance, pairs


def run_tiny(fixture):
    root, manifest, selection, config, provenance, pairs = fixture
    return fc.score_format(
        config,
        pairs,
        selection,
        root,
        root,
        [torch.tensor(list(b"train only train only"))],
        manifest,
        root,
        root / "report.json",
        provenance,
        checkpoint_origin="synthetic",
    )


def test_pipeline_and_identical_negative_control(tiny_control):
    result = run_tiny(tiny_control)
    for row in result["runs"]:
        assert row["pooled"]["changed"]["model"]["raw"]["count"] == 5
        assert row["pooled"]["identical"]["model"]["max_abs_pair_difference"] == 0.0
        assert row["pooled"]["identical"]["bigram"]["max_abs_pair_difference"] == 0.0
    assert result["aggregate"]["changed"]["arms"]["sinusoidal"]["projection_effect"]["n"] == 2
    with pytest.raises(ValueError, match="already exists"):
        run_tiny(tiny_control)


def test_changed_pair_manifest_rejected_before_forward(tiny_control, monkeypatch):
    tiny_control[-1]["documents"][0]["pairs"]["changed"][0]["target"] += 1
    monkeypatch.setattr(fc, "last_target_losses", lambda *a: pytest.fail("unexpected forward"))
    with pytest.raises(ValueError, match="pair manifest"):
        run_tiny(tiny_control)


def test_last_checkpoint_change_rejected_before_forward(tiny_control, monkeypatch):
    root, _, selection, *_ = tiny_control
    row = json.loads(selection.read_text())["runs"][-1]
    path = root / row["checkpoint"]
    saved = torch.load(path, weights_only=True)
    saved["model"]["output.bias"][0] += 1
    torch.save(saved, path)
    monkeypatch.setattr(fc, "last_target_losses", lambda *a: pytest.fail("unexpected forward"))
    with pytest.raises(ValueError, match="tensor identity"):
        run_tiny(tiny_control)
