"""Модели и их «бэкенды» (кодирование входа + сборка модели). Единственный модуль с классами nn.Module.

ngram - компактная модель из ТЗ (<1 млн параметров, считается на CPU);
qwen  - небольшой Qwen с LoRA и классификационной головой, помещается в 8 ГБ (RTX 4060).
Оптимизаторы обучают одни и те же обучаемые параметры: LoRA-адаптеры + голова (всё в FP32).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .features import ngram_ids, qwen_text, value_features
from .labels import N_CLASSES


class NgramBag(nn.Module):
    """EmbeddingBag по хешам n-грамм -> concat(признаки значений) -> Linear(hidden) -> ReLU -> Dropout -> Linear."""

    def __init__(self, buckets: int, emb_dim: int, hidden: int, dropout: float, n_value_feats: int = 3):
        super().__init__()
        self.emb = nn.EmbeddingBag(buckets, emb_dim, mode="mean")
        self.net = nn.Sequential(nn.Linear(emb_dim + n_value_feats, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, N_CLASSES))

    def forward(self, ids, offsets, vf):
        return self.net(torch.cat([self.emb(ids, offsets), vf], dim=1))


class QwenClassifier(nn.Module):
    """Qwen (заморожен, bf16) + LoRA (FP32) -> усреднение по токенам -> Dropout -> Linear."""

    def __init__(self, qcfg: dict):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        dtype = getattr(torch, qcfg["base_dtype"]) if torch.cuda.is_available() else torch.float32
        base = AutoModel.from_pretrained(qcfg["name"], torch_dtype=dtype)
        lcfg = LoraConfig(r=qcfg["lora_r"], lora_alpha=qcfg["lora_alpha"], lora_dropout=qcfg["lora_dropout"],
                          target_modules=qcfg["lora_targets"], bias="none")
        self.backbone = get_peft_model(base, lcfg)
        for p in self.backbone.parameters():     # обучаемое (LoRA) держим в FP32
            if p.requires_grad:
                p.data = p.data.float()
        hidden = base.config.hidden_size
        self.drop = nn.Dropout(qcfg["head_dropout"])
        self.head = nn.Linear(hidden, N_CLASSES)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled.float()))


class NgramBackend:
    name = "ngram"

    def __init__(self, cfg: dict):
        self.c = cfg["model"]["ngram"]
        self.micro_batch = None
        self._cache: dict = {}

    def encode_one(self, header: str, values: str):
        k = (header, values)
        it = self._cache.get(k)
        if it is None:
            it = (ngram_ids(header, self.c["buckets"], self.c["ngram_min"], self.c["ngram_max"]), value_features(values))
            self._cache[k] = it
        return it

    def collate(self, items, device):
        lens = [len(it[0]) for it in items]
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]])
        return {"ids": torch.tensor([i for it in items for i in it[0]], dtype=torch.long, device=device),
                "offsets": torch.tensor(offsets, dtype=torch.long, device=device),
                "vf": torch.tensor([it[1] for it in items], dtype=torch.float32, device=device)}

    def build_model(self, seed: int):
        torch.manual_seed(seed)
        return NgramBag(self.c["buckets"], self.c["emb_dim"], self.c["hidden"], self.c["dropout"])


class QwenBackend:
    name = "qwen"

    def __init__(self, cfg: dict):
        from transformers import AutoTokenizer

        self.q = cfg["model"]["qwen"]
        self.micro_batch = self.q.get("micro_batch", 16)
        self.tok = AutoTokenizer.from_pretrained(self.q["name"])
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        self._cache: dict = {}

    def encode_one(self, header: str, values: str):
        k = (header, values)
        it = self._cache.get(k)
        if it is None:
            it = self.tok(qwen_text(header, values), truncation=True, max_length=self.q["max_len"])["input_ids"]
            self._cache[k] = it
        return it

    def collate(self, items, device):
        L = max(len(x) for x in items)
        ids = torch.full((len(items), L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(items), L), dtype=torch.long)
        for i, x in enumerate(items):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}

    def build_model(self, seed: int):
        torch.manual_seed(seed)
        return QwenClassifier(self.q)


def make_backend(cfg: dict, name: str | None = None):
    name = name or cfg["model"]["backend"]
    if name == "ngram":
        return NgramBackend(cfg)
    if name == "qwen":
        return QwenBackend(cfg)
    raise KeyError(name)


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
