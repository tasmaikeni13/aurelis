"""Reproducible byte-level corpora and diagnostic sequence generators."""

from __future__ import annotations

import hashlib
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
from torch import Tensor


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILES = {
    "wikitext-2-raw-v1": {
        "train": [("train-00000-of-00001.parquet", "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7")],
        "validation": [("validation-00000-of-00001.parquet", "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c")],
        "test": [("test-00000-of-00001.parquet", "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91")],
    },
    "wikitext-103-raw-v1": {
        "train": [
            ("train-00000-of-00002.parquet", "74da360f23826045b3e6ac6375411fdb15f003030aa74f2596ed08b857cb9212"),
            ("train-00001-of-00002.parquet", "ba090ac30dbf5461e8dcbdd1a1b8e6f3cf9c2c756d64f0c1220450acd514f720"),
        ],
        "validation": [("validation-00000-of-00001.parquet", "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c")],
        "test": [("test-00000-of-00001.parquet", "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91")],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and sha256(path) == expected_sha256:
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    urllib.request.urlretrieve(url, temporary)
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}: {actual}")
    temporary.replace(path)


def wikitext_bytes(
    cache: Path,
    configuration: str,
    split: str,
    *,
    maximum_bytes: int | None = None,
) -> bytes:
    """Download pinned WikiText parquet shards and return UTF-8 corpus bytes."""

    if configuration not in WIKITEXT_FILES:
        raise ValueError(f"unknown WikiText configuration: {configuration}")
    if split not in WIKITEXT_FILES[configuration]:
        raise ValueError(f"unknown split: {split}")
    output = bytearray()
    for filename, checksum in WIKITEXT_FILES[configuration][split]:
        local = cache / configuration / filename
        url = (
            "https://huggingface.co/datasets/Salesforce/wikitext/resolve/"
            f"{WIKITEXT_REVISION}/{configuration}/{filename}"
        )
        _download(url, local, checksum)
        table = pq.read_table(local, columns=["text"])
        for text in table.column("text").to_pylist():
            if text:
                output.extend(text.encode("utf-8", errors="replace"))
                if not text.endswith("\n"):
                    output.extend(b"\n")
            if maximum_bytes is not None and len(output) >= maximum_bytes:
                return bytes(output[:maximum_bytes])
    return bytes(output if maximum_bytes is None else output[:maximum_bytes])


@dataclass(frozen=True)
class DiagnosticExample:
    task: str
    prompt: bytes
    target: bytes

    @property
    def sequence(self) -> bytes:
        return self.prompt + self.target + b"\n"


TASKS = (
    "associative_recall",
    "variable_tracking",
    "repeated_name_recall",
    "exact_value_retrieval",
    "in_context_regression",
    "multi_hop",
)


def _associative(rng: random.Random, long: bool) -> DiagnosticExample:
    count = 12 if long else 5
    keys = rng.sample(list("abcdefghjkmnpqrstuvwxyz"), count)
    values = rng.sample(range(10, 100), count)
    query = rng.randrange(count)
    prompt = ";".join(f"K[{key}]={value}" for key, value in zip(keys, values, strict=True))
    return DiagnosticExample("associative_recall", f"{prompt};GET[{keys[query]}]=".encode(), str(values[query]).encode())


def _variable(rng: random.Random, long: bool) -> DiagnosticExample:
    variables = list("wxyz")
    count = 16 if long else 7
    assignments: list[tuple[str, int]] = []
    current: dict[str, int] = {}
    for _ in range(count):
        name = rng.choice(variables)
        value = rng.randrange(10, 100)
        assignments.append((name, value))
        current[name] = value
    query = rng.choice(list(current))
    prompt = ";".join(f"{name}:={value}" for name, value in assignments)
    return DiagnosticExample("variable_tracking", f"{prompt};VALUE({query})=".encode(), str(current[query]).encode())


def _repeated_name(rng: random.Random, long: bool) -> DiagnosticExample:
    names = ["Ada", "Bo", "Cy", "Di"]
    colors = ["red", "blue", "green", "gold", "black", "white"]
    count = 14 if long else 6
    facts: list[tuple[str, str]] = []
    latest: dict[str, str] = {}
    for _ in range(count):
        name, color = rng.choice(names), rng.choice(colors)
        facts.append((name, color))
        latest[name] = color
    query = rng.choice(list(latest))
    prompt = " ".join(f"{name} has {color}." for name, color in facts)
    return DiagnosticExample("repeated_name_recall", f"{prompt} Latest color for {query}: ".encode(), latest[query].encode())


def _exact_value(rng: random.Random, long: bool) -> DiagnosticExample:
    count = 10 if long else 4
    names = rng.sample(["AX", "BY", "CZ", "DU", "EV", "FW", "GX", "HY", "IZ", "JU", "KV", "LW"], count)
    values = [f"{rng.randrange(10000, 100000)}" for _ in names]
    query = rng.randrange(count)
    prompt = "|".join(f"{name}#{value}" for name, value in zip(names, values, strict=True))
    return DiagnosticExample("exact_value_retrieval", f"{prompt}|LOOKUP({names[query]})#".encode(), values[query].encode())


def _regression(rng: random.Random, long: bool) -> DiagnosticExample:
    slope = rng.choice([-3, -2, -1, 1, 2, 3])
    intercept = rng.randrange(-5, 6)
    count = 8 if long else 4
    xs = rng.sample(range(1, 15), count + 1)
    pairs = ";".join(f"x={x},y={slope*x+intercept}" for x in xs[:-1])
    target = slope * xs[-1] + intercept
    return DiagnosticExample("in_context_regression", f"{pairs};x={xs[-1]},y=".encode(), str(target).encode())


def _multihop(rng: random.Random, long: bool) -> DiagnosticExample:
    hops = 6 if long else 3
    nodes = rng.sample(list("abcdefghjkmnpqrstuvwxyz"), hops + 4)
    chain = nodes[: hops + 1]
    distractors = [(nodes[-3], nodes[-2]), (nodes[-2], nodes[-1])]
    edges = list(zip(chain[:-1], chain[1:], strict=True)) + distractors
    rng.shuffle(edges)
    prompt = ";".join(f"{left}>{right}" for left, right in edges)
    return DiagnosticExample("multi_hop", f"{prompt};CHASE({chain[0]},{hops})=".encode(), chain[-1].encode())


GENERATORS = {
    "associative_recall": _associative,
    "variable_tracking": _variable,
    "repeated_name_recall": _repeated_name,
    "exact_value_retrieval": _exact_value,
    "in_context_regression": _regression,
    "multi_hop": _multihop,
}


def diagnostic_examples(
    seed: int,
    count_per_task: int,
    *,
    long: bool = False,
) -> list[DiagnosticExample]:
    rng = random.Random(seed)
    examples = [
        GENERATORS[task](rng, long)
        for task in TASKS
        for _ in range(count_per_task)
    ]
    rng.shuffle(examples)
    return examples


def diagnostic_corpus(seed: int, minimum_bytes: int) -> bytes:
    output = bytearray()
    generation = 0
    while len(output) < minimum_bytes:
        for example in diagnostic_examples(seed + 104729 * generation, 64):
            output.extend(example.sequence)
        generation += 1
    return bytes(output[:minimum_bytes])


def bytes_to_tensor(data: bytes, device: torch.device | str) -> Tensor:
    # bytearray owns writable storage, avoiding torch.frombuffer's read-only warning.
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).long().to(device)


def sample_token_batch(
    natural: Tensor,
    diagnostic: Tensor,
    batch_size: int,
    sequence_length: int,
    diagnostic_fraction: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample matched next-byte sequences from natural and diagnostic streams."""

    if not 0 <= diagnostic_fraction <= 1:
        raise ValueError("diagnostic_fraction must be in [0,1]")
    width = sequence_length + 1
    choose_diagnostic = torch.rand(batch_size, generator=generator, device=natural.device) < diagnostic_fraction
    natural_offsets = torch.randint(
        natural.numel() - width + 1,
        (batch_size,),
        generator=generator,
        device=natural.device,
    )
    diagnostic_offsets = torch.randint(
        diagnostic.numel() - width + 1,
        (batch_size,),
        generator=generator,
        device=natural.device,
    )
    positions = torch.arange(width, device=natural.device)
    natural_rows = natural[natural_offsets[:, None] + positions]
    diagnostic_rows = diagnostic[diagnostic_offsets[:, None] + positions]
    rows = torch.where(choose_diagnostic[:, None], diagnostic_rows, natural_rows)
    return rows[:, :-1], rows[:, 1:]
