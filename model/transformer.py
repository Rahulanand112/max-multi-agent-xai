"""MAX-LM: the decoder-only transformer.

    tokens -> token embedding + positional embedding
           -> N x TransformerBlock (pre-LN attention + feed-forward)
           -> final LayerNorm
           -> LM head (weights tied to the token embedding)
           -> next-token logits

Config A parameter count, calculated:
    token embedding   4096 x 128                =   524,288
    positional         128 x 128                =    16,384
    4 blocks @ 197,760                          =   791,040
    final layernorm                             =       256
    LM head (tied -- adds nothing)              =         0
    ------------------------------------------------------
                                                  1,331,968
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock
from .init import expected_init_loss, init_weights


class MAXTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        context_length: int,
        positional: str = "learned",
        attn_bias: bool = False,
        ffn_bias: bool = True,
        tie_embeddings: bool = True,
        dropout: float = 0.0,
        init_std: float = 0.02,
        scale_residual: bool = True,
        use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if positional != "learned":
            raise NotImplementedError(
                f"positional={positional!r} is not implemented yet. "
                "Config A uses learned absolute positions; RoPE arrives with Config B."
            )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_length = context_length
        self.tie_embeddings = tie_embeddings

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                context_length=context_length,
                attn_bias=attn_bias,
                ffn_bias=ffn_bias,
                dropout=dropout,
                use_sdpa=use_sdpa,
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # --- RANDOM INITIALISATION. Every weight is sampled here. ---
        init_weights(self, std=init_std, n_layers=n_layers, scale_residual=scale_residual)

        # Tie AFTER initialising, so the shared tensor is sampled exactly once.
        # nn.Module.parameters() de-duplicates, so the tied head adds 0 params.
        if tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.register_buffer(
            "position_ids",
            torch.arange(context_length).unsqueeze(0),
            persistent=False,
        )

    # ------------------------------------------------------------- inspection

    def num_parameters(self, trainable_only: bool = False) -> int:
        params = self.parameters()  # de-duplicates tied weights
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def parameter_breakdown(self) -> list[tuple[str, int]]:
        """(component, parameter count) rows, for the Review 1 table."""
        rows: list[tuple[str, int]] = [
            ("token embedding", self.tok_emb.weight.numel()),
            ("positional embedding", self.pos_emb.weight.numel()),
        ]
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        rows.append((f"transformer block (x1 of {self.n_layers})", per_block))
        rows.append((f"all {self.n_layers} blocks", per_block * self.n_layers))
        rows.append(("final layernorm", sum(p.numel() for p in self.ln_f.parameters())))
        rows.append((
            "LM head (tied)" if self.tie_embeddings else "LM head",
            0 if self.tie_embeddings else self.lm_head.weight.numel(),
        ))
        rows.append(("TOTAL", self.num_parameters()))
        return rows

    def expected_init_loss(self) -> float:
        """ln(vocab_size) -- what step-0 loss must equal. See model/init.py."""
        return expected_init_loss(self.vocab_size)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_hidden: bool = False,
    ):
        batch, seq_len = idx.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context length {self.context_length}"
            )

        positions = self.position_ids[:, :seq_len]
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        if return_hidden:
            # the pooled hidden states the V5 coordinator will consume
            return logits, loss, hidden
        return logits, loss

    # ------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. temperature=0 gives greedy decoding."""
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.context_length:]
            logits, _ = self(context)
            logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_token], dim=1)
            if eos_id is not None and bool((next_token == eos_id).all()):
                break
        return idx


def build_model(cfg, use_sdpa: bool = False) -> MAXTransformer:
    """Construct MAX-LM from a config object (see configs/model_a.yaml)."""
    m = cfg.model
    init = cfg.get("init", {})
    return MAXTransformer(
        vocab_size=m.vocab_size,
        d_model=m.d_model,
        n_layers=m.n_layers,
        n_heads=m.n_heads,
        d_ff=m.d_ff,
        context_length=m.context_length,
        positional=m.get("positional", "learned"),
        attn_bias=m.get("attn_bias", False),
        ffn_bias=m.get("ffn_bias", True),
        tie_embeddings=m.get("tie_embeddings", True),
        dropout=m.get("dropout", 0.0),
        init_std=init.get("std", 0.02),
        scale_residual=init.get("scale_residual", True),
        use_sdpa=use_sdpa,
    )
