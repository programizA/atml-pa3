"""The SigLIP -> connector -> Qwen3 vision-language model.

A training example is one chat turn:

    <|im_start|>user\n [196 visual tokens] \n<question or caption instruction><|im_end|>\n
    <|im_start|>assistant\n<think>\n\n</think>\n\n<answer><|im_end|>

The visual tokens are not words. We tokenize the chat with 196 placeholder tokens (<|image_pad|>),
look up ordinary text embeddings for everything else, then overwrite the placeholder positions
with the connector's outputs Z(I) = s * g(F(I)). The language model never sees the placeholder's
own embedding. Labels are -100 everywhere except the answer/caption tokens and the closing
<|im_end|>, so the loss only grades what the assistant says. Positions are ordinary 0..L-1 over
the real (non-padding) tokens, so visual tokens occupy 196 positions like any other tokens.

For "no image" inputs the placeholders (and the newline after them) are simply left out and the
masks and positions are rebuilt for the shorter sequence; the text is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from task2 import config as C


class Connector(nn.Module):
    """Two-layer MLP: 768 -> 1024 -> GELU -> 1024, then a fixed calibration scale s."""

    def __init__(self, d_in=C.D_VIS, d_out=C.D_LM):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_out, d_out)
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, f):
        return self.fc2(self.act(self.fc1(f))) * self.scale


class Encoder:
    """Turns (prompt text, target text, has_image) into token ids and labels."""

    def __init__(self, tok, n_vis=C.N_PATCH):
        self.tok = tok
        self.n_vis = n_vis
        self.img_id = tok.convert_tokens_to_ids("<|image_pad|>")
        self.eot = tok.convert_tokens_to_ids("<|im_end|>")
        assert self.img_id != tok.unk_token_id

    def cap(self, text, n):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= n else self.tok.decode(ids[:n])

    def prompt_content(self, ex):
        if ex["kind"] == "caption":
            return C.CAPTION_PROMPT
        return self.cap(ex["question"], C.Q_CAP) + C.SHORT_ANSWER_SUFFIX

    def encode(self, ex, image=True, with_target=True):
        """ex: {"kind", "question"?, "target"}. Returns dict(ids, labels, n_target_content)."""
        content = self.prompt_content(ex)
        if image:
            content = "<|image_pad|>" * self.n_vis + "\n" + content
        prompt = self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        assert sum(t == self.img_id for t in p_ids) == (self.n_vis if image else 0)
        if not with_target:
            return {"ids": p_ids, "labels": [-100] * len(p_ids), "n_target": 0}
        cap = C.CAPTION_CAP if ex["kind"] == "caption" else C.ANSWER_CAP
        t_ids = self.tok(ex["target"], add_special_tokens=False)["input_ids"][:cap]
        ids = p_ids + t_ids + [self.eot]
        labels = [-100] * len(p_ids) + t_ids + [self.eot]
        return {"ids": ids, "labels": labels, "n_target": len(t_ids)}


def pad_batch(encs, pad_id, side="right"):
    L = max(len(e["ids"]) for e in encs)
    B = len(encs)
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    lab = torch.full((B, L), -100, dtype=torch.long)
    att = torch.zeros((B, L), dtype=torch.long)
    for i, e in enumerate(encs):
        n = len(e["ids"])
        sl = slice(0, n) if side == "right" else slice(L - n, L)
        ids[i, sl] = torch.tensor(e["ids"])
        lab[i, sl] = torch.tensor(e["labels"])
        att[i, sl] = 1
    pos = (att.cumsum(1) - 1).clamp(min=0)
    return ids, lab, att, pos


def build_inputs(lm, connector, ids, feats, img_id):
    """Text embeddings with the placeholder rows replaced by connector outputs.

    feats: (n_images_with_image_tokens, 196, 768) in the same order as the rows of `ids` that
    contain placeholders. Gradients flow into the connector through the replaced rows."""
    emb = lm.get_input_embeddings()(ids)
    mask = ids == img_id
    if mask.any():
        vis = connector(feats.to(ids.device, connector.fc1.weight.dtype))     # (n, 196, 1024)
        emb = emb.clone()
        emb[mask] = vis.reshape(-1, vis.shape[-1]).to(emb.dtype)
    return emb


def _ce_chunk(h, t, W):
    return F.cross_entropy(F.linear(h, W).float(), t, reduction="none")


def token_nll(lm, emb, att, pos, labels, chunk=2048):
    """Per-token NLL at supervised positions: returns (nll flat, row index of each token)."""
    h = lm.model(inputs_embeds=emb, attention_mask=att, position_ids=pos).last_hidden_state[:, :-1]
    tgt = labels[:, 1:]
    m = tgt != -100
    rows = torch.nonzero(m, as_tuple=True)[0]
    hs, ts = h[m], tgt[m]
    W = lm.lm_head.weight
    parts = []
    for i in range(0, ts.numel(), chunk):
        if torch.is_grad_enabled():
            parts.append(checkpoint(_ce_chunk, hs[i:i + chunk], ts[i:i + chunk], W, use_reentrant=False))
        else:
            parts.append(_ce_chunk(hs[i:i + chunk], ts[i:i + chunk], W))
    nll = torch.cat(parts) if parts else hs.new_zeros(0, dtype=torch.float32)
    return nll, rows


def per_example_mean(nll, rows, B):
    s = torch.zeros(B, device=nll.device, dtype=torch.float32).index_add_(0, rows, nll)
    n = torch.zeros(B, device=nll.device, dtype=torch.float32).index_add_(0, rows, torch.ones_like(nll))
    return s / n.clamp(min=1), n


@torch.no_grad()
def calibrate_scale(lm, connector, encoder, train_examples, feats, n_tokens_sample=None):
    """Rule: s = median L2 norm of Qwen's text embeddings over the tokens of training prompts and
    targets / median L2 norm of the freshly initialised connector's outputs on training images.
    Training data only. Afterwards visual and text tokens enter the LM at the same typical size."""
    E = lm.get_input_embeddings().weight
    tok_ids = []
    for ex in train_examples:
        e = encoder.encode(ex, image=False)
        tok_ids += [t for t in e["ids"]]
    tok_ids = torch.tensor(tok_ids, device=E.device)
    text_norm = E[tok_ids].float().norm(dim=-1).median().item()
    connector.scale.fill_(1.0)
    vis = connector(feats.to(E.device).float())
    vis_norm = vis.norm(dim=-1).median().item()
    s = text_norm / vis_norm
    connector.scale.fill_(s)
    return {"rule": "median ||text embedding|| / median ||connector output at init||",
            "text_embedding_median_norm": text_norm, "connector_init_median_norm": vis_norm, "scale": s,
            "n_text_tokens": int(tok_ids.numel()), "n_images": int(feats.shape[0])}
