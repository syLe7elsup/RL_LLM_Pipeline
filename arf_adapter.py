"""Adapter that loads artifacts from the sister LLM_RL pipeline so our
explanation-side optimizations (polarity-aware K-candidate selection, gold
buffer few-shot, embedding verifier, snapshot loading, etc.) can run on the
DQN-on-ARF setup without re-implementing stages 1-5.

Source:
    https://github.com/Papillon-Xiang/LLM_RL  (stages 1-5)

LLM_RL produces these artifacts under ``data/outputs/``:

    dqn_model.pt                # full QNetwork state_dict (input_dim->128->64->3)
    sae_model.pt                # SparseAutoencoder state_dict (in=64, latent=128)
    {train,val,test}_hidden_states.pt
        {hidden(N,64), actions(N), q_values(N,3), rewards(N),
         patient_ids[N], time_steps[N]}
    {train,val,test}_sae_latents.pt
        {latents(N,128), patient_ids[N], time_steps[N]}
    feature_columns.json   (under data/processed/)

Our pipeline (`bb_pipeline.*`) was originally written for binary outcome
prediction. The simplest faithful mapping for the DQN setup is:

    binary label y = (action_taken == IMV)

i.e. "did the clinician escalate to mechanical ventilation at this step?"
The action distribution in our data is dominated by 0 (standard oxygen) vs 2
(IMV), with action=1 (HFNC/NIV) used in <1% of rows, so this binary view
captures most of the policy signal.

Returns a dict shaped like our pipeline's ``state`` (see ``state_io.py``):
    {
        "X":            (n, F)   raw or scaled state features
        "y":            (n,)     binary label
        "tr_idx", "va_idx":      train/val row indices into X / Z
        "scaler":       {"mean", "std"}
        "model":        a tiny shim object with .predict_proba(X)
        "sae":          shim with .encode(H) returning the saved SAE latents
        "H_tr", "H_va": (n, 64)
        "Z_tr", "Z_va": (n, 128)  pre-computed by LLM_RL stage 5
        "evidence":     None  (caller must call collect_evidence on Z_tr)
        "concept_labels", "polarities": None until LLM1 / polarity stages run
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# Default location on the user's laptop. Override via load_arf_state(...).
DEFAULT_ARTIFACT_DIR = Path("/Users/abigail/Downloads/llmproject/LLM_RL/data/outputs")
DEFAULT_PROCESSED_DIR = Path("/Users/abigail/Downloads/llmproject/LLM_RL/data/processed")


@dataclass
class _DqnShim:
    """Wraps the LLM_RL QNetwork with the same .predict_proba interface our
    bb_pipeline expects (input -> [P_low, P_high])."""

    qnet: torch.nn.Module
    imv_action: int = 2

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Treat softmax(Q-values) as the policy distribution; collapse it to
        a binary "did/didn't choose IMV" by summing over non-IMV vs IMV."""
        self.qnet.eval()
        q = self.qnet(x)                       # (B, 3)
        probs = torch.softmax(q, dim=-1)       # policy distribution
        p_imv = probs[..., self.imv_action]
        p_other = 1.0 - p_imv
        return torch.stack([p_other, p_imv], dim=-1)  # [P_low, P_high]

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        self.qnet.eval()
        with torch.no_grad():
            _, h = self.qnet(x, return_hidden=True)
        return h

    def parameters(self):
        return self.qnet.parameters()

    def to(self, *args, **kwargs):
        self.qnet.to(*args, **kwargs)
        return self


@dataclass
class _SaeShim:
    """Encoder-only shim wrapping LLM_RL's SparseAutoencoder. We expose the
    same .encode method bb_pipeline.sae.TopKSAE has."""

    sae: torch.nn.Module
    k: int = 0   # 0 means no top-k mask (LLM_RL uses L1 sparsity, not top-k)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.sae.eval()
        return self.sae.encode(x)

    def parameters(self):
        return self.sae.parameters()


# ---------------------------------------------------------------------------


def _build_qnetwork(input_dim: int, n_actions: int = 3,
                    hidden1: int = 128, hidden2: int = 64):
    """Mirror of LLM_RL/src/dqn_model.py:QNetwork."""
    import torch.nn as nn

    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden1)
            self.fc2 = nn.Linear(hidden1, hidden2)
            self.out = nn.Linear(hidden2, n_actions)
            self.relu = nn.ReLU()

        def forward(self, x, return_hidden: bool = False):
            h1 = self.relu(self.fc1(x))
            h2 = self.relu(self.fc2(h1))
            q = self.out(h2)
            if return_hidden:
                return q, h2
            return q

    return QNetwork()


def _build_sae(input_dim: int = 64, latent_dim: int = 128):
    """Mirror of LLM_RL/src/sae_model.py:SparseAutoencoder."""
    import torch.nn as nn
    import torch.nn.functional as F

    class SparseAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(input_dim, latent_dim)
            self.decoder = nn.Linear(latent_dim, input_dim)

        def forward(self, x):
            z = torch.relu(self.encoder(x))
            return self.decoder(z), z

        def encode(self, x):
            return torch.relu(self.encoder(x))

    return SparseAutoencoder()


# ---------------------------------------------------------------------------


def load_arf_state(
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    *,
    imv_action: int = 2,
    device: str | None = None,
) -> dict:
    """Load LLM_RL artifacts and return a state dict compatible with
    ``bb_pipeline.state_io.load_pipeline_state``'s shape (so the rest of the
    pipeline's notebooks / scripts can consume it transparently).

    Args:
        artifact_dir:  where stage-3..5 outputs live (data/outputs/).
        processed_dir: where stage-2 outputs live (data/processed/).
        imv_action:    integer action that maps to "IMV" (default 2 in LLM_RL).
        device:        torch device for the QNetwork shim (auto-detect if None).
    """
    from .blackbox import auto_device

    artifact_dir = Path(artifact_dir)
    processed_dir = Path(processed_dir)
    device = device or auto_device()

    # ----- feature names + scaler (from stage 2) -----
    with open(processed_dir / "feature_columns.json") as f:
        feature_names = json.load(f)
    F_dim = len(feature_names)

    # ----- DQN -----
    qnet = _build_qnetwork(input_dim=F_dim)
    qnet.load_state_dict(torch.load(artifact_dir / "dqn_model.pt", map_location=device))
    qnet.to(device).eval()
    model = _DqnShim(qnet=qnet, imv_action=imv_action)

    # ----- SAE -----
    sae_inner = _build_sae(input_dim=64, latent_dim=128)
    sae_inner.load_state_dict(torch.load(artifact_dir / "sae_model.pt", map_location=device))
    sae_inner.to(device).eval()
    sae = _SaeShim(sae=sae_inner)

    # ----- transitions (already standardized + imputed by stage 2) -----
    train_t = torch.load(processed_dir / "train_transitions.pt", weights_only=False)
    val_t   = torch.load(processed_dir / "val_transitions.pt",   weights_only=False)
    # Concatenate train + val (we keep train for evidence, val for test inputs).
    X_tr = train_t["states"]    # already-normalized, shape (n_tr, F)
    X_va = val_t["states"]
    a_tr = train_t["actions"]
    a_va = val_t["actions"]

    # Binary label
    y_tr = (a_tr == imv_action).astype(np.int64)
    y_va = (a_va == imv_action).astype(np.int64)

    X = np.concatenate([X_tr, X_va], axis=0).astype(np.float32)
    y = np.concatenate([y_tr, y_va], axis=0)
    tr_idx = np.arange(len(X_tr))
    va_idx = np.arange(len(X_tr), len(X_tr) + len(X_va))

    # Scaler is identity here because stage 2 already standardized.
    scaler = {
        "mean": np.zeros((1, F_dim), dtype=np.float32),
        "std":  np.ones((1, F_dim), dtype=np.float32),
    }

    # ----- precomputed hidden states + SAE latents (stage 4 + 5) -----
    # LLM_RL only saves train+test SAE latents (no val). We re-encode val on
    # the fly using the loaded SAE so all three splits are available.
    H_tr_dict = torch.load(artifact_dir / "train_hidden_states.pt", weights_only=False)
    H_va_dict = torch.load(artifact_dir / "val_hidden_states.pt",   weights_only=False)
    Z_tr_dict = torch.load(artifact_dir / "train_sae_latents.pt",   weights_only=False)
    H_tr = np.asarray(H_tr_dict["hidden"], dtype=np.float32)
    H_va = np.asarray(H_va_dict["hidden"], dtype=np.float32)
    Z_tr = np.asarray(Z_tr_dict["latents"], dtype=np.float32)
    with torch.no_grad():
        H_va_t = torch.from_numpy(H_va).to(device)
        Z_va = sae_inner.encode(H_va_t).cpu().numpy().astype(np.float32)

    return {
        "feature_names": feature_names,
        "X":      X,
        "y":      y,
        "tr_idx": tr_idx,
        "va_idx": va_idx,
        "scaler": scaler,
        "model":  model,
        "sae":    sae,
        "H_tr": H_tr, "H_va": H_va,
        "Z_tr": Z_tr, "Z_va": Z_va,
        # bb_pipeline downstream expects these to optionally be filled later.
        "evidence":       None,
        "concept_labels": None,
        "verifier_acc":   None,
        "polarities":     None,
        # Extras for ARF-specific analysis
        "actions_train":  a_tr,
        "actions_val":    a_va,
        "q_values_train": np.asarray(H_tr_dict["q_values"], dtype=np.float32),
        "q_values_val":   np.asarray(H_va_dict["q_values"], dtype=np.float32),
        "rewards_train":  np.asarray(H_tr_dict["rewards"], dtype=np.float32),
        "rewards_val":    np.asarray(H_va_dict["rewards"], dtype=np.float32),
    }
