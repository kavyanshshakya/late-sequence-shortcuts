"""Dataset and Example dataclass."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    premises: str
    question: str
    answer: str
    label: int
    cot: Optional[str]
    shortcut_available: bool
    reasoning_type: str
    n_hops: int
    n_distractors: int
    shortcut_cue: Optional[str] = None


class ReasoningDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len=320, include_cot=False):
        self.examples = examples
        ids_l, labels, masks = [], [], []
        for ex in examples:
            text = f"{ex.premises} <sep> {ex.question} <ans> {ex.answer}"
            ids = tokenizer.encode(text)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len - 1] + [2]
            m = [1] * len(ids)
            pad = max_seq_len - len(ids)
            ids += [0] * pad
            m += [0] * pad
            ids_l.append(ids)
            labels.append(ex.label)
            masks.append(m)
        self.input_ids = torch.tensor(ids_l, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.masks[i],
        }
