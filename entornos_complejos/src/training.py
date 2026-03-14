"""Wrappers de compatibilidad sobre la implementación consolidada.

Este módulo preserva la API histórica de entrenamiento, pero delega la lógica
real en ``control_aprox_utils.py`` para evitar mantener dos stacks distintos.
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .control_aprox_utils import (
    final_window_seed_means,
    set_global_seed,
    smooth,
    train_dqn_cartpole as _train_dqn_cartpole,
    train_dqn_tetris as _train_dqn_tetris,
    train_sarsa_cartpole as _train_sarsa_cartpole,
    train_sarsa_tetris as _train_sarsa_tetris,
)

__all__ = [
    "set_global_seed",
    "cartpole_obs",
    "tetris_obs_flat",
    "train_sarsa_cartpole",
    "train_sarsa_tetris",
    "train_dqn_cartpole",
    "train_dqn_tetris",
    "run_multiseed",
    "summary_stats",
]


def cartpole_obs(obs) -> np.ndarray:
    """Conversión ligera a ``float32`` mantenida por compatibilidad."""
    return np.asarray(obs, dtype=np.float32)


def tetris_obs_flat(obs) -> np.ndarray:
    """Aplana la observación de Tetris para código legado."""
    if isinstance(obs, dict):
        board = obs.get("board", list(obs.values())[0])
    else:
        board = obs
    return np.asarray(board, dtype=np.float32).flatten()


def train_sarsa_cartpole(
    n_episodes: int = 600,
    seed: int = 0,
    alpha: float = 3e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    fourier_order: int = 2,
) -> List[float]:
    """Wrapper legado sobre la implementación consolidada de SARSA-CartPole."""
    return _train_sarsa_cartpole(
        n_episodes=n_episodes,
        seed=seed,
        alpha=alpha,
        gamma=gamma,
        eps_start=epsilon_start,
        eps_end=epsilon_end,
        eps_decay=epsilon_decay,
        fourier_order=fourier_order,
        return_agent=False,
    )


def train_sarsa_tetris(
    n_episodes: int = 200,
    seed: int = 0,
    alpha: float = 1e-2,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.99,
    max_steps: int = 2000,
) -> List[float]:
    """Wrapper legado sobre la implementación consolidada de SARSA-Tetris."""
    return _train_sarsa_tetris(
        n_episodes=n_episodes,
        seed=seed,
        alpha=alpha,
        gamma=gamma,
        eps_start=epsilon_start,
        eps_end=epsilon_end,
        eps_decay=epsilon_decay,
        max_steps=max_steps,
        return_agent=False,
    )


def train_dqn_cartpole(
    n_episodes: int = 600,
    seed: int = 0,
    lr: float = 5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.9995,
    buffer_size: int = 20_000,
    batch_size: int = 64,
    target_update_freq: int = 200,
    hidden_size: int = 128,
    n_layers: int = 2,
    learn_every: int = 1,
) -> List[float]:
    """Wrapper legado sobre la implementación consolidada de DQN-CartPole."""
    return _train_dqn_cartpole(
        n_episodes=n_episodes,
        seed=seed,
        lr=lr,
        gamma=gamma,
        eps_start=epsilon_start,
        eps_end=epsilon_end,
        eps_decay=epsilon_decay,
        buf_size=buffer_size,
        batch=batch_size,
        target_freq=target_update_freq,
        hidden=hidden_size,
        n_layers=n_layers,
        train_freq=learn_every,
        return_agent=False,
    )


def train_dqn_tetris(
    n_episodes: int = 200,
    seed: int = 0,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.999,
    buffer_size: int = 20_000,
    batch_size: int = 64,
    target_update_freq: int = 300,
    hidden_size: int = 256,
    n_layers: int = 2,
    max_steps: int = 2000,
) -> List[float]:
    """Wrapper legado sobre la implementación consolidada de DQN-Tetris."""
    return _train_dqn_tetris(
        n_episodes=n_episodes,
        seed=seed,
        lr=lr,
        gamma=gamma,
        eps_start=epsilon_start,
        eps_end=epsilon_end,
        eps_decay=epsilon_decay,
        buf_size=buffer_size,
        batch=batch_size,
        target_freq=target_update_freq,
        hidden=hidden_size,
        n_layers=n_layers,
        max_steps=max_steps,
        return_agent=False,
    )


def run_multiseed(
    train_fn: Callable,
    seeds: List[int],
    **kwargs,
) -> np.ndarray:
    """Ejecuta ``train_fn`` para varias semillas y apila los retornos."""
    return np.asarray([train_fn(seed=seed, **kwargs) for seed in seeds], dtype=float)


def summary_stats(returns_matrix: np.ndarray, window: int = 50) -> dict:
    """Resumen ligero para código legado basado en medias por semilla."""
    returns_matrix = np.asarray(returns_matrix, dtype=float)
    smoothed = np.asarray([smooth(r, window) for r in returns_matrix], dtype=float)
    mean_curve = smoothed.mean(axis=0)
    std_curve = smoothed.std(axis=0)
    return {
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "final_mean": float(final_window_seed_means(returns_matrix, window).mean()),
        "final_std": float(final_window_seed_means(returns_matrix, window).std()),
        "best_episode": int(np.argmax(mean_curve)) if mean_curve.size else 0,
    }
