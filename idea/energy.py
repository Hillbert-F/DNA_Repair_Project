"""Energy calculation utilities for IDEA."""

from __future__ import annotations

import json
from pathlib import Path


ENERGY_CALCULATION_VERSION = "idea-energy-v1"


class EnergyError(ValueError):
    """Raised when gamma and phi inputs cannot be combined."""


def _np():
    import numpy as np

    return np


def load_gamma(path: str | Path):
    np = _np()
    path = Path(path)
    return np.loadtxt(
        path,
        dtype=complex,
        converters={0: lambda s: complex(s.decode().replace("+-", "-"))},
    )


def load_phi(path: str | Path):
    np = _np()
    phi = np.loadtxt(path, dtype=float)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    return phi


def calculate_energy(gamma, phi):
    np = _np()
    gamma = np.asarray(gamma)
    phi = np.asarray(phi)
    if gamma.ndim != 1:
        gamma = gamma.reshape(-1)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    if phi.shape[1] != gamma.shape[0]:
        raise EnergyError(
            f"Feature dimension mismatch: gamma has {gamma.shape[0]} values, "
            f"phi has {phi.shape[1]} columns."
        )
    energy = phi @ gamma
    if np.iscomplexobj(energy):
        if not np.allclose(energy.imag, 0.0, atol=1e-8):
            raise EnergyError("Energy contains non-zero imaginary values.")
        energy = energy.real
    return energy


def calculate_energy_files(gamma_path: str | Path, phi_path: str | Path, out_dir: str | Path) -> Path:
    np = _np()
    gamma_path = Path(gamma_path).resolve()
    phi_path = Path(phi_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gamma = load_gamma(gamma_path)
    phi = load_phi(phi_path)
    energy = calculate_energy(gamma, phi)

    energy_path = out_dir / "Energy_mg.txt"
    np.savetxt(energy_path, energy, fmt="%f", delimiter="\n")
    manifest = {
        "version": ENERGY_CALCULATION_VERSION,
        "gamma": str(gamma_path),
        "phi": str(phi_path),
        "n_sequences": int(phi.shape[0]),
        "n_features": int(phi.shape[1]),
        "energy": str(energy_path),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return energy_path
