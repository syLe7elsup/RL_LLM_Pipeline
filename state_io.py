"""Save and load the expensive intermediate state of a pipeline run.

The LLM1 + verifier phase takes ~17 min on Qwen 3B. If you want to iterate on
just the LLM2 / Judge / polarity / loss-weight side, you don't want to re-run
that. Snapshot once, reload many times.

Layout (relative to a chosen ``out_dir``):
    out_dir/
        config.json
        traj.npz                # raw POLAR trajectories + label + scaler
        mlp_state_dict.pt       # blackbox weights
        sae_state_dict.pt       # SAE weights
        H_tr.npy / H_va.npy     # cached MLP hidden reps
        Z_tr.npy / Z_va.npy     # cached SAE latents
        evidence.pkl            # {fid: FeatureEvidence}
        concepts.json           # {fid: concept string}
        verifier_acc.json       # {fid: float}
        polarities.pkl          # {fid: FeaturePolarity}
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch


def save_pipeline_state(
    out_dir: str | os.PathLike,
    *,
    config: dict,
    traj: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    scaler: dict,
    model,
    sae,
    H_tr: np.ndarray,
    H_va: np.ndarray,
    Z_tr: np.ndarray,
    Z_va: np.ndarray,
    evidence: dict,
    concept_labels: dict,
    verifier_records: dict,
    polarities: dict | None = None,
) -> Path:
    """Persist everything needed to resume from the LLM2 / Judge stage."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "config.json").open("w") as f:
        json.dump({k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                   for k, v in config.items()}, f, indent=2)

    np.savez(
        out / "traj.npz",
        traj=traj, X=X, y=y, tr_idx=tr_idx, va_idx=va_idx,
        scaler_mean=scaler["mean"], scaler_std=scaler["std"],
    )
    torch.save(model.state_dict(), out / "mlp_state_dict.pt")
    torch.save(sae.state_dict(), out / "sae_state_dict.pt")
    np.save(out / "H_tr.npy", H_tr)
    np.save(out / "H_va.npy", H_va)
    np.save(out / "Z_tr.npy", Z_tr)
    np.save(out / "Z_va.npy", Z_va)

    with (out / "evidence.pkl").open("wb") as f:
        pickle.dump(evidence, f)
    with (out / "concepts.json").open("w") as f:
        json.dump({str(k): v for k, v in concept_labels.items()}, f, indent=2)

    acc_dict = {str(k): float(v.accuracy) for k, v in verifier_records.items()}
    with (out / "verifier_acc.json").open("w") as f:
        json.dump(acc_dict, f, indent=2)

    if polarities is not None:
        with (out / "polarities.pkl").open("wb") as f:
            pickle.dump(polarities, f)

    print(f"saved snapshot to {out}")
    return out


def load_pipeline_state(in_dir: str | os.PathLike, *, mlp_class, sae_class, device=None):
    """Reload everything saved by ``save_pipeline_state``.

    Returns a dict with the same keys you passed in. ``mlp_class`` and
    ``sae_class`` are the constructors (passed in to keep this module free of
    cycle-prone imports).
    """
    from .blackbox import auto_device
    in_dir = Path(in_dir)
    device = device or auto_device()

    with (in_dir / "config.json").open() as f:
        config = json.load(f)

    npz = np.load(in_dir / "traj.npz")
    traj, X, y = npz["traj"], npz["X"], npz["y"]
    tr_idx, va_idx = npz["tr_idx"], npz["va_idx"]
    scaler = {"mean": npz["scaler_mean"], "std": npz["scaler_std"]}

    H_tr = np.load(in_dir / "H_tr.npy")
    H_va = np.load(in_dir / "H_va.npy")
    Z_tr = np.load(in_dir / "Z_tr.npy")
    Z_va = np.load(in_dir / "Z_va.npy")

    model = mlp_class(in_dim=X.shape[1], hidden_dim=int(config.get("mlp_hidden", 64)))
    model.load_state_dict(torch.load(in_dir / "mlp_state_dict.pt", map_location=device))
    model.to(device).eval()

    sae = sae_class(in_dim=H_tr.shape[1], latent_dim=int(config.get("sae_latent", 32)),
                    k=int(config.get("sae_k", 8)))
    sae.load_state_dict(torch.load(in_dir / "sae_state_dict.pt", map_location=device))
    sae.to(device).eval()

    with (in_dir / "evidence.pkl").open("rb") as f:
        evidence = pickle.load(f)
    with (in_dir / "concepts.json").open() as f:
        concept_labels = {int(k): v for k, v in json.load(f).items()}
    with (in_dir / "verifier_acc.json").open() as f:
        verifier_acc = {int(k): float(v) for k, v in json.load(f).items()}

    polarities = None
    pol_path = in_dir / "polarities.pkl"
    if pol_path.exists():
        with pol_path.open("rb") as f:
            polarities = pickle.load(f)

    return dict(
        config=config, traj=traj, X=X, y=y, tr_idx=tr_idx, va_idx=va_idx,
        scaler=scaler, model=model, sae=sae,
        H_tr=H_tr, H_va=H_va, Z_tr=Z_tr, Z_va=Z_va,
        evidence=evidence, concept_labels=concept_labels,
        verifier_acc=verifier_acc, polarities=polarities,
    )
