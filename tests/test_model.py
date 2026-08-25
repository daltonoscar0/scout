"""Model tests.

The causality test is the one that earns its keep. A leak from future positions
would not crash or even look wrong during training -- the loss would simply fall
faster than it should, and the embeddings would encode what happened next rather
than how a player plays. It is only detectable by construction.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="model tests need the `train` extra")

from scout import vocab  # noqa: E402
from scout.model import (  # noqa: E402
    ModelConfig,
    PossessionLM,
    cosine_schedule,
    loss,
    select_device,
)


def batch(n=2, seq=32):
    tokens = torch.zeros(n, seq, 5, dtype=torch.long)
    for field, size in enumerate(vocab.FIELD_SIZES):
        tokens[..., field] = torch.randint(0, size, (n, seq))
    return tokens


@pytest.fixture(scope="module")
def tiny():
    return PossessionLM(ModelConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, seq_len=32))


# --- shape and size -------------------------------------------------------


def test_both_configs_stay_inside_the_twenty_million_cap():
    assert PossessionLM(ModelConfig.small()).n_parameters() <= 20_000_000
    assert PossessionLM(ModelConfig.medium()).n_parameters() <= 20_000_000


def test_the_small_config_is_the_smaller_one():
    assert (
        PossessionLM(ModelConfig.small()).n_parameters()
        < PossessionLM(ModelConfig.medium()).n_parameters()
    )


def test_a_head_per_field_with_the_right_width(tiny):
    logits = tiny(batch())
    assert len(logits) == len(vocab.FIELD_SIZES)
    for head, size in zip(logits, vocab.FIELD_SIZES, strict=True):
        assert head.shape[-1] == size


def test_hidden_states_are_one_vector_per_position(tiny):
    hidden = tiny.hidden_states(batch(n=2, seq=32))
    assert hidden.shape == (2, 32, tiny.config.d_model)


def test_an_indivisible_head_count_is_rejected_at_construction():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=100, n_heads=8)


# --- causality ------------------------------------------------------------


def test_a_position_cannot_see_the_future(tiny):
    tokens = batch(n=1, seq=16)
    tiny.eval()
    with torch.no_grad():
        before = tiny.hidden_states(tokens)[0, :8].clone()
        # Rewrite everything after position 8 and re-run.
        changed = tokens.clone()
        changed[0, 8:, 0] = (changed[0, 8:, 0] + 5) % vocab.ACTION_SIZE
        after = tiny.hidden_states(changed)[0, :8]
    assert torch.allclose(before, after, atol=1e-5)


def test_changing_the_past_does_change_the_present(tiny):
    # The mirror of the test above: if nothing propagated forward either, the
    # causal test would pass on a model that ignores its input entirely.
    tokens = batch(n=1, seq=16)
    tiny.eval()
    with torch.no_grad():
        before = tiny.hidden_states(tokens)[0, 8:].clone()
        changed = tokens.clone()
        changed[0, :8, 0] = (changed[0, :8, 0] + 5) % vocab.ACTION_SIZE
        after = tiny.hidden_states(changed)[0, 8:]
    assert not torch.allclose(before, after, atol=1e-5)


# --- embeddings -----------------------------------------------------------


def test_the_pad_row_of_every_table_starts_at_zero(tiny):
    for table in tiny.embed.tables:
        assert torch.count_nonzero(table.weight[0]) == 0


def test_the_five_fields_contribute_independently(tiny):
    # If a field were dropped from the sum, changing it alone would be a no-op.
    base = batch(n=1, seq=8)
    tiny.eval()
    for field, size in enumerate(vocab.FIELD_SIZES):
        changed = base.clone()
        changed[0, :, field] = (changed[0, :, field] + 1) % size
        with torch.no_grad():
            same = torch.allclose(tiny.embed(base), tiny.embed(changed), atol=1e-6)
        assert not same, f"field {vocab.FIELD_NAMES[field]} does not affect the embedding"


# --- loss -----------------------------------------------------------------


def test_padding_is_excluded_from_the_loss(tiny):
    tokens = batch(n=2, seq=16)
    full = torch.ones(2, 16)
    half = torch.ones(2, 16)
    half[:, 8:] = 0
    logits = tiny(tokens)
    a, _ = loss(logits, tokens, full)
    b, _ = loss(logits, tokens, half)
    assert not torch.isclose(a, b)


def test_an_all_padding_batch_does_not_divide_by_zero(tiny):
    tokens = batch(n=1, seq=8)
    value, _ = loss(tiny(tokens), tokens, torch.zeros(1, 8))
    assert torch.isfinite(value)


def test_loss_reports_a_component_per_predicted_field(tiny):
    tokens = batch(n=1, seq=8)
    _, parts = loss(tiny(tokens), tokens, torch.ones(1, 8))
    for name in ("action", "start_zone", "end_delta", "flags", "dt_bucket"):
        assert name in parts


def test_gradients_reach_every_parameter(tiny):
    tokens = batch(n=2, seq=16)
    value, _ = loss(tiny(tokens), tokens, torch.ones(2, 16))
    tiny.zero_grad()
    value.backward()
    missing = [
        name
        for name, p in tiny.named_parameters()
        if p.grad is None or torch.count_nonzero(p.grad) == 0
    ]
    # Padding rows legitimately receive no gradient; nothing else should be dead.
    assert all("position" in name or "tables" in name for name in missing), missing


# --- schedule and device --------------------------------------------------


def test_the_schedule_warms_up_then_decays():
    warm = [cosine_schedule(s, warmup=10, total=100, peak=1.0) for s in range(10)]
    assert warm == sorted(warm)
    assert cosine_schedule(50, warmup=10, total=100, peak=1.0) < cosine_schedule(
        10, warmup=10, total=100, peak=1.0
    )


def test_the_schedule_never_reaches_zero():
    assert cosine_schedule(100, warmup=10, total=100, peak=1.0) > 0


def test_an_explicit_device_is_honoured():
    assert select_device("cpu").type == "cpu"
