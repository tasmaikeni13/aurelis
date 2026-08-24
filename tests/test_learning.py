from __future__ import annotations

import torch
import pytest

from csm import (
    BoundedScalarGate,
    EpisodicMemoryModel,
    FP64GaussMarkovMemory,
    geometry_metrics,
    orthogonality_penalty,
)


def test_bounded_scalar_gate_respects_support() -> None:
    gate = BoundedScalarGate(3, 5, minimum=0.01, maximum=4.0)
    values = gate(torch.randn(11, 3))
    assert torch.all(values >= 0.01)
    assert torch.all(values <= 4.0)


def test_batched_learned_csm_matches_scalar_reference() -> None:
    torch.manual_seed(41)
    model = EpisodicMemoryModel(5, 3, 4, 3, 7, epsilon=0.2, kind="csm").double()
    raw_keys = torch.randn(2, 4, 5, dtype=torch.float64)
    raw_values = torch.randn(2, 4, 3, dtype=torch.float64)
    raw_queries = torch.randn(2, 2, 5, dtype=torch.float64)
    beta = 0.2 + torch.rand(2, 4, dtype=torch.float64)
    output = model(raw_keys, raw_values, raw_queries, beta=beta)
    assert output.state is not None and output.uncertainty is not None
    for batch in range(2):
        memory = FP64GaussMarkovMemory(4, 3, epsilon=0.2)
        state = memory.undiscounted_state(
            output.keys[batch], output.values[batch], beta[batch]
        )
        latent, uncertainty = memory.read_many_with_confidence(
            state, output.queries[batch]
        )
        torch.testing.assert_close(
            output.prediction[batch], model.decode(latent), rtol=1e-11, atol=1e-11
        )
        torch.testing.assert_close(
            output.uncertainty[batch], uncertainty, rtol=1e-11, atol=1e-11
        )


def test_fixed_random_features_freeze_encoders_but_not_decoder() -> None:
    model = EpisodicMemoryModel(
        6, 4, 5, 4, 8, epsilon=0.1, fixed_random_features=True
    )
    assert all(not parameter.requires_grad for parameter in model.key_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.value_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.output_decoder.parameters())
    raw = torch.randn(3, 2, 6)
    torch.testing.assert_close(model.encode_keys(raw), model.encode_queries(raw))


def test_learned_keys_and_queries_share_a_coordinate_chart() -> None:
    model = EpisodicMemoryModel(5, 3, 4, 3, 8, epsilon=0.1)
    assert model.key_encoder is model.query_encoder
    sample = torch.randn(2, 3, 5)
    torch.testing.assert_close(model.encode_keys(sample), model.encode_queries(sample))


def test_independent_query_map_is_an_explicit_ablation() -> None:
    model = EpisodicMemoryModel(
        5, 3, 4, 3, 8, epsilon=0.1, shared_key_query=False
    )
    assert model.key_encoder is not model.query_encoder


def test_geometry_metrics_and_regularizer_on_orthogonal_keys() -> None:
    keys = torch.eye(5).unsqueeze(0).repeat(3, 1, 1)
    metrics = geometry_metrics(keys, epsilon=0.1)
    assert metrics["effective_rank"] == 5.0
    assert metrics["mean_absolute_pairwise_cosine"] == 0.0
    assert metrics["minimum_singular_value"] == 1.0
    torch.testing.assert_close(orthogonality_penalty(keys), torch.tensor(0.0))


def test_effective_capacity_uses_reachable_rank() -> None:
    keys = torch.eye(6, dtype=torch.float64)[:3].unsqueeze(0)
    metrics = geometry_metrics(keys, epsilon=0.1)
    assert metrics["effective_rank"] == pytest.approx(3.0)
    assert metrics["effective_capacity_fraction"] == pytest.approx(1.0)


def test_sequential_write_uses_beta_for_both_statistics() -> None:
    model = EpisodicMemoryModel(3, 2, 3, 2, 5, epsilon=0.1)
    reference = torch.zeros(2, 3)
    state = model.initial_state(2, reference)
    key = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    value = torch.randn(2, 2)
    beta = torch.tensor([0.25, 2.0])
    decay = torch.tensor([1.0, 0.5])
    updated = model.write_state(state, key, value, beta, decay)
    torch.testing.assert_close(
        updated.S, beta[:, None, None] * key.unsqueeze(-1) * key.unsqueeze(-2)
    )
    torch.testing.assert_close(
        updated.C, beta[:, None, None] * value.unsqueeze(-1) * key.unsqueeze(-2)
    )
