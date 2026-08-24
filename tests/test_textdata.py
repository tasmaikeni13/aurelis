from __future__ import annotations

import torch

from csm import TASKS, bytes_to_tensor, diagnostic_corpus, diagnostic_examples, sample_token_batch


def test_diagnostic_examples_cover_every_task_and_are_reproducible() -> None:
    first = diagnostic_examples(42, 3)
    second = diagnostic_examples(42, 3)
    assert first == second
    assert {example.task for example in first} == set(TASKS)
    assert all(example.prompt and example.target and example.sequence.endswith(b"\n") for example in first)


def test_long_diagnostics_are_longer() -> None:
    short = diagnostic_examples(7, 4)
    long = diagnostic_examples(7, 4, long=True)
    assert sum(len(item.prompt) for item in long) > sum(len(item.prompt) for item in short)


def test_diagnostic_corpus_meets_exact_requested_size() -> None:
    assert len(diagnostic_corpus(1, 10_003)) == 10_003


def test_sample_batch_has_shifted_targets_and_both_sources() -> None:
    natural = bytes_to_tensor(bytes(range(100)), "cpu")
    diagnostic = bytes_to_tensor(bytes(range(100, 200)), "cpu")
    generator = torch.Generator().manual_seed(8)
    inputs, targets = sample_token_batch(natural, diagnostic, 32, 8, 0.5, generator)
    assert inputs.shape == targets.shape == (32, 8)
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])
    assert (inputs[:, 0] < 100).any()
    assert (inputs[:, 0] >= 100).any()
