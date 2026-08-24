from __future__ import annotations

import pytest
import torch

from csm import TinyDecoderLM


def model(kind: str) -> TinyDecoderLM:
    torch.manual_seed(901)
    return TinyDecoderLM(
        vocabulary_size=64,
        width=32,
        layers=2,
        heads=4,
        feedforward_width=64,
        architecture=kind,  # type: ignore[arg-type]
        csm_key_dimension=4,
        csm_epsilon=0.2,
        local_window=4,
    ).eval()


@pytest.mark.parametrize("kind", ["transformer", "csm", "hybrid", "recurrent"])
def test_decoder_variants_are_causal_and_finite(kind: str) -> None:
    decoder = model(kind)
    tokens = torch.randint(64, (2, 8))
    changed = tokens.clone()
    changed[:, 5:] = torch.randint(64, changed[:, 5:].shape)
    first = decoder(tokens)
    second = decoder(changed)
    assert first.shape == (2, 8, 64)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[:, :5], second[:, :5], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["transformer", "csm", "hybrid", "recurrent"])
def test_incremental_logits_match_full_forward(kind: str) -> None:
    decoder = model(kind)
    tokens = torch.randint(64, (2, 7))
    expected = decoder(tokens)
    states = None
    pieces = []
    for position in range(tokens.shape[1]):
        logits, states = decoder.step(tokens[:, position], states, position)
        pieces.append(logits)
    actual = torch.cat(pieces, dim=1)
    tolerance = 3e-4 if kind in ("csm", "hybrid") else 2e-5
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_csm_state_is_context_independent_while_attention_cache_grows() -> None:
    csm = model("csm")
    transformer = model("transformer")
    assert csm.recurrent_state_bytes(2, 16) == csm.recurrent_state_bytes(2, 1024)
    assert transformer.recurrent_state_bytes(2, 1024) == 64 * transformer.recurrent_state_bytes(2, 16)


def test_generate_preserves_prompt_and_length() -> None:
    decoder = model("hybrid")
    prompt = torch.randint(64, (2, 5))
    generated = decoder.generate(prompt, 3)
    assert generated.shape == (2, 8)
    torch.testing.assert_close(generated[:, :5], prompt)
