"""The parameter-update restrictions compared in Task 1.

Full FT and L2-SP train the model's own weights. LoRA and OPLoRA freeze them and add a small
correction to each targeted linear layer:

    output = W0 x + (alpha / r) * B (A x)

A (r x d_in) squeezes the input down to r numbers, B (d_out x r) maps them back up. B starts at
zero, so at step 0 the model is exactly the original. OPLoRA additionally projects both factors
away from the starting weight's strongest directions before they are used:

    B_eff = B - U_k (U_k^T B)      (remove the part of B's columns lying in W0's top-k output space)
    A_eff = A - (A V_k) V_k^T      (remove the part of A's rows lying in W0's top-k input space)

where W0 = U S V^T is the singular value decomposition and U_k, V_k are its first k columns.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from task1 import config as C


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank correction (optionally OPLoRA-projected)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, init_seed: int,
                 Uk: torch.Tensor | None = None, Vk: torch.Tensor | None = None):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        # Same initialisation as PEFT's default: A ~ kaiming_uniform(a=sqrt(5)), i.e. uniform in
        # +-1/sqrt(d_in); B = 0. A dedicated CPU generator per module makes LoRA-16 and OPLoRA-16
        # start from bit-identical unprojected factors.
        g = torch.Generator().manual_seed(init_seed)
        bound = 1.0 / math.sqrt(d_in)
        A = (torch.rand(r, d_in, generator=g, dtype=torch.float32) * 2 - 1) * bound
        dev = base.weight.device
        self.lora_A = nn.Parameter(A.to(dev))                                   # FP32
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.float32, device=dev))
        self.dropout = nn.Dropout(dropout)
        self.projected = Uk is not None
        if self.projected:
            self.register_buffer("Uk", Uk.to(dev, torch.float32), persistent=False)
            self.register_buffer("Vk", Vk.to(dev, torch.float32), persistent=False)

    def effective_factors(self):
        """(A_eff, B_eff) in FP32. For plain LoRA these are just A and B. Computed with autocast
        off so the projection is exact to FP32 precision rather than FP16."""
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            A, B = self.lora_A, self.lora_B
            if self.projected:
                B = B - self.Uk @ (self.Uk.T @ B)
                A = A - (A @ self.Vk) @ self.Vk.T
        return A, B

    def delta_weight(self) -> torch.Tensor:
        A, B = self.effective_factors()
        return self.scale * (B @ A)

    def forward(self, x):
        A, B = self.effective_factors()
        if not torch.is_autocast_enabled(x.device.type):   # e.g. FP32 diagnostics, merge checks
            A, B = A.to(x.dtype), B.to(x.dtype)
        return self.base(x) + F.linear(F.linear(self.dropout(x), A), B) * self.scale


def target_linears(model, targets=C.LORA_TARGETS):
    """(name, parent module, attribute, linear) for every targeted projection, in a fixed order."""
    out = []
    for name, mod in model.named_modules():
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                out.append((f"{name}.{t}", mod, t, child))
    return out


@torch.no_grad()
def top_singular_bases(W: torch.Tensor, k: int):
    """Leading k left/right singular vectors of a weight matrix, computed once in FP32.

    On the GPU we ask for the classic (more accurate) gesvd solver, and then re-orthonormalise
    the k kept vectors with a QR step. QR keeps exactly the same subspace, but removes the small
    loss of orthogonality (~1e-4) that GPU SVD solvers leave, so the projection really is exact
    to FP32 precision."""
    Wf = W.float()
    kw = {"driver": "gesvd"} if Wf.is_cuda else {}
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False, **kw)
    Uk, _ = torch.linalg.qr(U[:, :k])
    Vk, _ = torch.linalg.qr(Vh[:k, :].T)
    return Uk.contiguous(), Vk.contiguous(), S


def add_adapters(model, r, alpha, dropout=C.LORA_DROPOUT, protect_k=None, seed=C.SEED, targets=C.LORA_TARGETS):
    """Freeze the model and wrap every target linear with LoRALinear. With protect_k set, compute
    the OPLoRA bases from each starting weight. Returns overhead measurements."""
    for p in model.parameters():
        p.requires_grad_(False)
    stats = {"basis_seconds": 0.0, "basis_bytes": 0, "n_modules": 0}
    for idx, (name, parent, attr, lin) in enumerate(target_linears(model, targets)):
        Uk = Vk = None
        if protect_k:
            t0 = time.perf_counter()
            Uk, Vk, _ = top_singular_bases(lin.weight, protect_k)
            if Uk.is_cuda:
                torch.cuda.synchronize()
            stats["basis_seconds"] += time.perf_counter() - t0
            stats["basis_bytes"] += (Uk.numel() + Vk.numel()) * 4
        setattr(parent, attr, LoRALinear(lin, r, alpha, dropout, init_seed=seed * 1000 + idx, Uk=Uk, Vk=Vk))
        stats["n_modules"] += 1
    return stats


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


def adapter_state(model, projected=True):
    """Adapter weights to save. Raw factors always; for OPLoRA also the exported projected factors
    that inference and diagnostics use."""
    sd = {}
    for n, m in lora_modules(model):
        sd[f"{n}.lora_A"] = m.lora_A.detach().cpu().clone()
        sd[f"{n}.lora_B"] = m.lora_B.detach().cpu().clone()
        if projected and m.projected:
            A, B = m.effective_factors()
            sd[f"{n}.A_eff"] = A.detach().cpu().clone()
            sd[f"{n}.B_eff"] = B.detach().cpu().clone()
    return sd


def load_adapter_state(model, sd):
    for n, m in lora_modules(model):
        m.lora_A.data.copy_(sd[f"{n}.lora_A"])
        m.lora_B.data.copy_(sd[f"{n}.lora_B"])


@torch.no_grad()
def merge_into_fresh(base_model, sd, scale):
    """Fold a saved adapter into a fresh copy of the frozen backbone: W = W0 + scale * B_eff A_eff,
    computed in FP32 then cast back to the backbone dtype. Uses the projected factors for OPLoRA."""
    lins = {n: lin for n, _, _, lin in target_linears(base_model)}
    for n, lin in lins.items():
        if f"{n}.A_eff" in sd:
            A, B = sd[f"{n}.A_eff"], sd[f"{n}.B_eff"]
        else:
            A, B = sd[f"{n}.lora_A"], sd[f"{n}.lora_B"]
        dW = scale * (B.float() @ A.float())
        lin.weight.copy_((lin.weight.float() + dW.to(lin.weight.device)).to(lin.weight.dtype))
    return base_model


def unique_parameters(model):
    """Each parameter tensor once, even if tied (Qwen3-0.6B shares embedding and LM head)."""
    seen, out = set(), []
    for n, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append((n, p))
    return out


def count_params(model, trainable_only=True):
    return sum(p.numel() for _, p in unique_parameters(model) if (p.requires_grad or not trainable_only))


class L2SP:
    """Penalty Omega = 1/2 * sum_j (theta_j - theta0_j)^2 over every unique parameter.

    theta0 is stored in the checkpoint's original dtype (BF16 for Qwen3), which represents the
    starting weights exactly at half the memory of FP32. Instead of building an autograd graph
    for the penalty, its gradient lam * (theta - theta0) is added directly to each parameter's
    gradient after unscaling; that is mathematically the same as differentiating lam * Omega.
    """

    def __init__(self, params, theta0: dict, lam: float):
        self.params = params                        # list of (name, parameter)
        self.lam = lam
        dev = params[0][1].device
        t0 = time.perf_counter()
        self.ref = {n: theta0[n].to(dev, non_blocking=True) for n, _ in params}
        if dev.type == "cuda":
            torch.cuda.synchronize()
        self.transfer_seconds = time.perf_counter() - t0
        self.ref_bytes = sum(t.numel() * t.element_size() for t in self.ref.values())

    @torch.no_grad()
    def omega_and_add_grad(self, add_grad=True, chunk_elems=1 << 24):
        """Omega, and (optionally) grad += lam * (theta - theta0).

        Works through each tensor in slices of at most 16M numbers, so the temporary difference
        tensor stays at <= 64 MB. (A whole-tensor FP64 copy of the 156M-entry embedding matrix
        would need 1.2 GB, which is more than a T4 has left during Full FT.) Each slice's sum of
        squares is FP32; slices are added up in Python floats (FP64)."""
        omega = 0.0
        for n, p in self.params:
            pf, rf = p.view(-1), self.ref[n].view(-1)
            g = p.grad.view(-1) if (add_grad and p.grad is not None) else None
            for i in range(0, pf.numel(), chunk_elems):
                d = pf[i:i + chunk_elems].float() - rf[i:i + chunk_elems].float()
                omega += 0.5 * float(d.pow(2).sum())
                if g is not None:
                    g[i:i + chunk_elems].add_(d, alpha=self.lam)
                del d
        return omega


@torch.no_grad()
def displacement(params, theta0: dict, chunk_elems=1 << 24):
    """Full-model L2 distance ||theta - theta0||_2 over unique parameters (sliced, like Omega)."""
    tot = 0.0
    for n, p in params:
        pf, rf = p.view(-1), theta0[n].view(-1)
        for i in range(0, pf.numel(), chunk_elems):
            d = pf[i:i + chunk_elems].float() - rf[i:i + chunk_elems].to(p.device).float()
            tot += float(d.pow(2).sum())
    return math.sqrt(tot)
