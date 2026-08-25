"""The possession language model.

A small pre-norm causal transformer over factorized tokens. Config S is the
default at roughly 6.5M parameters, comfortably inside the spec's 20M cap;
config M at about 14.5M is the stretch, promoted only after S trains end to end.

Two design choices are forced by the hardware rather than by taste:

  * Ops are restricted to matmul, softmax, layer_norm, gelu and embedding, with
    a dense causal mask. Anything that falls back from MPS to CPU turns a four
    minute epoch into forty, and the fallback is silent by default. Training
    should be run once with PYTORCH_ENABLE_MPS_FALLBACK=0 to prove it does not
    happen.

  * There is no fused output head. The model predicts three fields -- action,
    start zone and end delta -- through three separate softmaxes, plus two
    auxiliary heads for flags and dt. A single fused softmax over the joint
    token space would be about 34 billion parameters.

This module imports torch, so nothing on the query path may import it. See
scout.cli for that boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from scout import vocab

# Fields the model predicts, with their loss weights. Action carries the most
# signal about what a player did; dt and flags are auxiliary context that help
# the representation without being interesting predictions in themselves.
PREDICTED_FIELDS = ("action", "start_zone", "end_delta", "flags", "dt_bucket")
LOSS_WEIGHTS = (1.0, 0.7, 0.5, 0.2, 0.2)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024
    seq_len: int = 256
    dropout: float = 0.0

    @classmethod
    def small(cls) -> ModelConfig:
        return cls()

    @classmethod
    def medium(cls) -> ModelConfig:
        return cls(d_model=384, n_layers=8, n_heads=12, d_ff=1536)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")


class FactorizedEmbedding(nn.Module):
    """Five small tables summed, rather than one table over the joint space."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.tables = nn.ModuleList(
            nn.Embedding(size, config.d_model, padding_idx=0) for size in vocab.FIELD_SIZES
        )
        self.position = nn.Embedding(config.seq_len, config.d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        # tokens: (batch, seq, 5)
        summed = self.tables[0](tokens[..., 0])
        for index in range(1, len(self.tables)):
            summed = summed + self.tables[index](tokens[..., index])
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return summed + self.position(positions)[None]


class Block(nn.Module):
    """Pre-norm block. Attention uses scaled_dot_product_attention with is_causal."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff_in = nn.Linear(config.d_model, config.d_ff)
        self.ff_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, dim = x.shape
        h = self.norm_attn(x)
        qkv = self.qkv(h).view(batch, seq, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch, seq, dim)
        x = x + self.proj(attended)

        h = self.norm_ff(x)
        h = self.ff_out(torch.nn.functional.gelu(self.ff_in(h)))
        return x + h


class PossessionLM(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.small()
        self.embed = FactorizedEmbedding(self.config)
        self.blocks = nn.ModuleList(Block(self.config) for _ in range(self.config.n_layers))
        self.norm = nn.LayerNorm(self.config.d_model)
        self.heads = nn.ModuleList(
            nn.Linear(self.config.d_model, size) for size in vocab.FIELD_SIZES
        )
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def hidden_states(self, tokens: Tensor) -> Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, tokens: Tensor) -> list[Tensor]:
        hidden = self.hidden_states(tokens)
        return [head(hidden) for head in self.heads]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def describe(self) -> str:
        total = self.n_parameters()
        return (
            f"PossessionLM d_model={self.config.d_model} layers={self.config.n_layers} "
            f"heads={self.config.n_heads} d_ff={self.config.d_ff}\n"
            f"  parameters: {total:,} ({total / 1e6:.2f}M) "
            f"-- spec cap is 20M, {'within' if total <= 20e6 else 'OVER'}"
        )

    def config_dict(self) -> dict:
        return asdict(self.config)


def loss(logits: list[Tensor], targets: Tensor, mask: Tensor) -> tuple[Tensor, dict[str, float]]:
    """Weighted cross-entropy over the predicted fields.

    `targets` is (batch, seq, 5) and `mask` is (batch, seq) marking real
    positions. Padding must be excluded explicitly rather than relying on
    ignore_index alone, because index 0 is a legitimate value in four of the
    five fields -- only the action field reserves 0 for PAD.
    """
    total = torch.zeros((), device=targets.device)
    parts: dict[str, float] = {}
    flat_mask = mask.reshape(-1)
    denominator = flat_mask.sum().clamp(min=1)

    for index, (name, weight) in enumerate(zip(PREDICTED_FIELDS, LOSS_WEIGHTS, strict=True)):
        prediction = logits[index][:, :-1].reshape(-1, logits[index].shape[-1])
        target = targets[:, 1:, index].reshape(-1)
        step_mask = mask[:, 1:].reshape(-1)
        per_token = torch.nn.functional.cross_entropy(prediction, target, reduction="none")
        field_loss = (per_token * step_mask).sum() / step_mask.sum().clamp(min=1)
        total = total + weight * field_loss
        parts[name] = float(field_loss.detach())

    parts["tokens"] = float(denominator)
    return total, parts


def select_device(prefer: str = "auto") -> torch.device:
    """Pick a device, preferring MPS on Apple Silicon.

    Falls back to CPU with a printed reason rather than silently, because the
    difference is roughly 5x on this workload and a silent fallback would look
    like the model simply being slow.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("MPS unavailable; falling back to CPU (expect roughly 5x slower training)")
    return torch.device("cpu")


def cosine_schedule(
    step: int, *, warmup: int, total: int, peak: float, floor: float = 0.1
) -> float:
    """Linear warmup then cosine decay to `floor` x peak."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
