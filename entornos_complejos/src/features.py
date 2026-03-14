"""Capa de compatibilidad para extractores de características.

La implementación única vive en ``control_aprox_utils.py``. Este módulo
mantiene los nombres históricos para evitar romper imports previos.
"""

from __future__ import annotations

import numpy as np

from .control_aprox_utils import (
    CARTPOLE_HIGH,
    CARTPOLE_LOW,
    FourierFeatures,
    N_TETRIS_FEAT,
    tetris_featurize,
)

CARTPOLE_OBS_LOW = CARTPOLE_LOW.astype(np.float32, copy=True)
CARTPOLE_OBS_HIGH = CARTPOLE_HIGH.astype(np.float32, copy=True)
N_TETRIS_FEATURES = N_TETRIS_FEAT

__all__ = [
    "CARTPOLE_OBS_LOW",
    "CARTPOLE_OBS_HIGH",
    "FourierFeatures",
    "cartpole_features",
    "tetris_features",
    "N_TETRIS_FEATURES",
]


def cartpole_features(order: int = 2) -> FourierFeatures:
    """Devuelve el extractor de Fourier listo para CartPole-v1."""
    return FourierFeatures(CARTPOLE_OBS_LOW, CARTPOLE_OBS_HIGH, order)


def tetris_features(obs) -> np.ndarray:
    """Mantiene el nombre histórico del extractor artesanal de Tetris."""
    return tetris_featurize(obs)
