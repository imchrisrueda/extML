"""
features.py
===========
Extractores de características para agentes con aproximación lineal.

Para CartPole-v1  → Base de Fourier de orden 2 sobre el estado normalizado.
Para Tetris       → Features artesanales del tablero (altura agregada, huecos,
                    irregularidad, líneas completas, altura máxima, desv. alturas).
"""

import numpy as np
from itertools import product


# ---------------------------------------------------------------------------
# CartPole: Base de Fourier
# ---------------------------------------------------------------------------

CARTPOLE_OBS_LOW  = np.array([-4.8, -3.5, -0.42, -3.5], dtype=np.float32)
CARTPOLE_OBS_HIGH = np.array([ 4.8,  3.5,  0.42,  3.5], dtype=np.float32)


def _fourier_coefficients(n_dims: int, order: int) -> np.ndarray:
    """Genera la matriz de coeficientes c para la base de Fourier coseno.
    Cada fila c_i tiene valores en {0,...,order}.
    Tamaño: (order+1)^n_dims × n_dims
    """
    indices = list(product(range(order + 1), repeat=n_dims))
    return np.array(indices, dtype=np.float32)          # (N_feat, n_dims)


class FourierFeatures:
    """Base de Fourier coseno para estados continuos (n-dimensional).

    phi_i(s) = cos(pi * c_i · s_norm)

    con s_norm ∈ [0,1]^n obtenido normalizando linealmente el estado.
    """

    def __init__(self, obs_low: np.ndarray, obs_high: np.ndarray, order: int = 2):
        self.low   = obs_low
        self.range = obs_high - obs_low + 1e-8
        self.C     = _fourier_coefficients(len(obs_low), order)   # (N, n)
        self.n_features = len(self.C)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_clipped = np.clip(obs, self.low, self.low + self.range)
        s_norm = (obs_clipped - self.low) / self.range            # [0,1]^n
        return np.cos(np.pi * self.C @ s_norm).astype(np.float32) # (N,)

    def alpha_scaling(self) -> np.ndarray:
        """Factores de escala de alpha recomendados por Konidaris et al.
        alpha_i = 1 / ||c_i||  (1 si ||c_i||=0)
        """
        norms = np.linalg.norm(self.C, axis=1)
        norms[norms == 0] = 1.0
        return 1.0 / norms


def cartpole_features(order: int = 2) -> FourierFeatures:
    """Devuelve el extractor de Fourier listo para CartPole-v1."""
    return FourierFeatures(CARTPOLE_OBS_LOW, CARTPOLE_OBS_HIGH, order)


# ---------------------------------------------------------------------------
# Tetris: Features artesanales del tablero
# ---------------------------------------------------------------------------

def get_board(obs) -> np.ndarray:
    """Extrae la matriz del tablero de la observacion (dict o ndarray).

    Acepta:
      - dict con clave 'board' o 'matrix' (formato de tetris_gymnasium)
      - ndarray directo (ya es el tablero)
    """
    if isinstance(obs, dict):
        board = obs.get("board", obs.get("matrix", None))
        if board is None:
            # Ultimo recurso: primer valor del dict
            board = list(obs.values())[0]
    else:
        board = np.asarray(obs)
    return np.asarray(board, dtype=np.float32)


def _column_heights(board: np.ndarray) -> np.ndarray:
    """Altura de cada columna (numero de celdas ocupadas desde la base)."""
    H, W = board.shape
    heights = np.zeros(W, dtype=np.float32)
    for c in range(W):
        col = board[:, c]
        occupied = np.where(col > 0)[0]
        if len(occupied) > 0:
            heights[c] = H - occupied[0]   # altura desde abajo
    return heights


def tetris_features(obs) -> np.ndarray:
    """
    Extrae 7 features normalizadas del tablero de Tetris.

    Features:
      0 - aggregate_height  (suma de alturas de columnas)
      1 - max_height        (altura maxima)
      2 - holes             (celdas vacias bajo celdas ocupadas)
      3 - bumpiness         (suma |h_i - h_{i+1}|)
      4 - std_heights       (desviacion estandar de alturas)
      5 - complete_lines    (filas completamente llenas)
      6 - wells             (numero de pozos laterales)

    Cada feature se normaliza al rango [0,1] con divisores heuristicos.
    """
    board = get_board(obs)
    H, W = board.shape
    heights   = _column_heights(board)

    # -- aggregate height [0, H*W] → [0,1]
    agg_h = float(heights.sum())

    # -- max height
    max_h = float(heights.max()) if W > 0 else 0.0

    # -- holes
    holes = 0
    for c in range(W):
        col   = board[:, c]
        found = False
        for cell in col:
            if cell > 0:
                found = True
            elif found:
                holes += 1

    # -- bumpiness
    bumpy = float(np.sum(np.abs(np.diff(heights))))

    # -- std
    std_h = float(np.std(heights))

    # -- complete lines
    complete = int(np.sum(np.all(board > 0, axis=1)))

    # -- wells (columna cuyas dos vecinas están al menos 2 más altas)
    wells = 0
    for c in range(W):
        left  = heights[c - 1] if c > 0 else H
        right = heights[c + 1] if c < W - 1 else H
        depth = min(left, right) - heights[c]
        if depth > 0:
            wells += int(depth)

    # Divisores para normalización heurística
    divs = np.array([H * W, H, H * W / 4, W * H, H, 4, H * W / 4],
                    dtype=np.float32) + 1e-8

    raw = np.array([agg_h, max_h, holes, bumpy, std_h, complete, wells],
                   dtype=np.float32)
    return np.clip(raw / divs, 0.0, 1.0)


N_TETRIS_FEATURES = 7
