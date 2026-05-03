"""Top-k Sparse Autoencoder (SAE).

Implements the standard top-k SAE recipe described in
    "Scaling and evaluating sparse autoencoders" (Gao et al., OpenAI 2024).

Architecture (no biases on encoder, tied or untied decoder, decoder weights
unit-norm):
    z = TopK( W_enc (x - b_pre) )           # k-sparse latent
    x_hat = W_dec z + b_pre                  # reconstruction

We use untied weights (separate W_enc and W_dec) since the dimensions are
small here and tying offers no real benefit. Decoder rows are unit-norm
constrained after each step to prevent the trivial scale-shrinking solution
where W_enc/W_dec collude to keep z small.

For our blackbox the input to the SAE is the 64-dim post-ReLU hidden state.
Default config (chosen for the MVP): latent=128 (2x overcomplete), k=4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """Keep top-k absolute values per row, zero the rest."""
    # We follow the common practice of selecting top-k by activation value
    # (post-ReLU style); since we apply ReLU before TopK, all activations are
    # >= 0 and we can use the raw values directly.
    if k >= z.shape[-1]:
        return z
    topk_vals, topk_idx = torch.topk(z, k=k, dim=-1)
    out = torch.zeros_like(z)
    out.scatter_(-1, topk_idx, topk_vals)
    return out


class TopKSAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, k: int):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(latent_dim, in_dim))
        self.W_dec = nn.Parameter(torch.empty(in_dim, latent_dim))
        self.b_pre = nn.Parameter(torch.zeros(in_dim))

        # Init: small Gaussian, then unit-norm decoder rows.
        nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5 ** 0.5)
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True) + 1e-8)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-TopK latent (post-ReLU)."""
        return F.relu((x - self.b_pre) @ self.W_enc.T)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """k-sparse latent."""
        return _topk_mask(self.encode_pre(x), self.k)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-norm each decoder column (one per latent feature)."""
        norm = self.W_dec.norm(dim=0, keepdim=True) + 1e-8
        self.W_dec.div_(norm)


@dataclass
class SAETrainResult:
    train_recon: list = field(default_factory=list)
    val_recon: list = field(default_factory=list)
    final_train_recon: float = 0.0
    final_val_recon: float = 0.0
    explained_variance: float = 0.0
    feature_density: np.ndarray | None = None  # frac of inputs that activate each feature
    dead_features: int = 0


def train_sae(
    H_train: np.ndarray,
    H_val: np.ndarray,
    latent_dim: int = 32,
    k: int = 8,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 256,
    aux_k: int = 8,
    aux_alpha: float = 1 / 32,
    dead_steps_threshold: int = 200,
    device: torch.device | None = None,
    seed: int = 0,
    log_every: int = 0,
) -> tuple[TopKSAE, SAETrainResult]:
    """Train a top-k SAE with the Gao-2024 auxiliary "dead feature" loss.

    A latent unit is considered "dead" if it has not been in the top-k for
    the last ``dead_steps_threshold`` training steps. We then take the
    top-``aux_k`` of those dead units' pre-activations, reconstruct the
    residual (x - x_hat) using only those, and add ``aux_alpha`` times that
    reconstruction loss. This nudges dead features toward becoming useful.
    """
    from .blackbox import auto_device  # late import to avoid cycle
    device = device or auto_device()
    torch.manual_seed(seed)

    in_dim = H_train.shape[1]
    sae = TopKSAE(in_dim=in_dim, latent_dim=latent_dim, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    H_tr = torch.from_numpy(H_train.astype(np.float32)).to(device)
    H_va = torch.from_numpy(H_val.astype(np.float32)).to(device)
    n = H_tr.shape[0]

    # Track per-feature steps since last firing.
    last_fired = torch.zeros(latent_dim, dtype=torch.long, device=device)
    step = 0

    train_recon, val_recon = [], []
    for epoch in range(epochs):
        sae.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x = H_tr[idx]
            z_pre = sae.encode_pre(x)              # (b, latent)
            z = _topk_mask(z_pre, sae.k)
            x_hat = sae.decode(z)
            loss = F.mse_loss(x_hat, x)

            fired = (z > 0).any(dim=0)             # (latent,) bool
            last_fired[fired] = step
            dead_mask = (step - last_fired) > dead_steps_threshold
            if aux_alpha > 0 and dead_mask.any():
                aux_k_eff = min(aux_k, int(dead_mask.sum().item()))
                if aux_k_eff > 0:
                    z_dead = z_pre.clone()
                    z_dead[:, ~dead_mask] = 0
                    z_dead = _topk_mask(z_dead, aux_k_eff)
                    residual = x - x_hat.detach()
                    aux_recon = sae.decode(z_dead) - sae.b_pre  # (b, in_dim); decode adds b_pre, undo it
                    loss = loss + aux_alpha * F.mse_loss(aux_recon, residual)

            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.normalize_decoder()
            epoch_loss += loss.item() * idx.numel()
            step += 1
        train_recon.append(epoch_loss / n)

        sae.eval()
        with torch.no_grad():
            x_hat_va, _ = sae(H_va)
            val_recon.append(F.mse_loss(x_hat_va, H_va).item())

        if log_every and ((epoch + 1) % log_every == 0):
            n_dead = int(((step - last_fired) > dead_steps_threshold).sum().item())
            print(
                f"  epoch {epoch+1:>3d}  train_recon={train_recon[-1]:.4f}  "
                f"val_recon={val_recon[-1]:.4f}  dead={n_dead}/{latent_dim}"
            )

    sae.eval()
    with torch.no_grad():
        z_va = sae.encode(H_va)
        x_hat_va = sae.decode(z_va)
        var_y = H_va.var()
        explained_var = 1.0 - F.mse_loss(x_hat_va, H_va).item() / (var_y + 1e-12)
        density = (z_va > 0).float().mean(dim=0).cpu().numpy()

    result = SAETrainResult(
        train_recon=train_recon,
        val_recon=val_recon,
        final_train_recon=train_recon[-1],
        final_val_recon=val_recon[-1],
        explained_variance=float(explained_var),
        feature_density=density,
        dead_features=int((density == 0).sum()),
    )
    return sae, result


@torch.no_grad()
def encode_all(sae: TopKSAE, H: np.ndarray, device: torch.device | None = None, batch_size: int = 1024) -> np.ndarray:
    """Run the SAE encoder on a batch of hidden representations and return the
    k-sparse latent z (n, latent_dim)."""
    from .blackbox import auto_device
    device = device or auto_device()
    sae.eval()
    H_t = torch.from_numpy(H.astype(np.float32)).to(device)
    out = []
    for i in range(0, H_t.shape[0], batch_size):
        out.append(sae.encode(H_t[i : i + batch_size]).cpu())
    return torch.cat(out, dim=0).numpy()
