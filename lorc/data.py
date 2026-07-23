from __future__ import annotations

import datasets
import torch
from torch import Tensor
from collections.abc import Iterator
from transformers import PreTrainedTokenizer


def load_minif2f(split: str = "train") -> list[str]:
    ds = datasets.load_dataset("openai/miniF2F", split=split)
    return [ex["text"] for ex in ds]


def load_wikipedia(n: int = 10000) -> list[str]:
    ds = datasets.load_dataset(
        "danbraunai/pile-uncopyrighted-tok-shuffled", split="train", streaming=True
    )
    texts: list[str] = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        texts.append(ex["text"])
    return texts


def domain_dataloader(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    seed: int = 42,
) -> Iterator[Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    n = len(texts)
    idx = torch.randperm(n, generator=g).tolist()
    pi = 0

    while True:
        batch: list[Tensor] = []
        for _ in range(batch_size):
            if pi >= n:
                pi = 0
                idx = torch.randperm(n, generator=g).tolist()
            text = texts[idx[pi]]
            pi += 1
            tokens = tokenizer(
                text, truncation=True, max_length=seq_len, return_tensors="pt"
            )["input_ids"][0]
            if tokens.size(0) < seq_len:
                pad = torch.full((seq_len - tokens.size(0),), tokenizer.pad_token_id, dtype=torch.long)
                tokens = torch.cat([tokens, pad])
            batch.append(tokens)
        yield torch.stack(batch)


def interleaved_dataloader(
    lean_texts: list[str],
    wiki_texts: list[str],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    ratio: float = 0.5,
    seed: int = 42,
) -> Iterator[tuple[Tensor, Tensor]]:
    g = torch.Generator()
    g.manual_seed(seed)
    n_lean = len(lean_texts)
    n_wiki = len(wiki_texts)
    idx_lean = torch.randperm(n_lean, generator=g).tolist()
    idx_wiki = torch.randperm(n_wiki, generator=g).tolist()
    pi_lean = pi_wiki = 0

    while True:
        input_ids_list: list[Tensor] = []
        domain_list: list[float] = []
        for _ in range(batch_size):
            if torch.rand(1, generator=g).item() < ratio:
                if pi_lean >= n_lean:
                    pi_lean = 0
                    idx_lean = torch.randperm(n_lean, generator=g).tolist()
                text = lean_texts[idx_lean[pi_lean]]
                pi_lean += 1
                domain = 1.0
            else:
                if pi_wiki >= n_wiki:
                    pi_wiki = 0
                    idx_wiki = torch.randperm(n_wiki, generator=g).tolist()
                text = wiki_texts[idx_wiki[pi_wiki]]
                pi_wiki += 1
                domain = 0.0
            tokens = tokenizer(
                text, truncation=True, max_length=seq_len, return_tensors="pt"
            )["input_ids"][0]
            if tokens.size(0) < seq_len:
                pad = torch.full((seq_len - tokens.size(0),), tokenizer.pad_token_id, dtype=torch.long)
                tokens = torch.cat([tokens, pad])
            input_ids_list.append(tokens)
            domain_list.append(domain)

        yield torch.stack(input_ids_list), torch.tensor(domain_list, dtype=torch.float32)
