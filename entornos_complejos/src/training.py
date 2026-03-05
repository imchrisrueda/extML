"""
training.py
===========
Bucles de entrenamiento y evaluación para los agentes SARSA y DQN.
Incluye también utilidades para fijar semillas y preprocesar observaciones.
"""

import random
from typing import Callable, Optional, List

import numpy as np
import torch
import gymnasium as gym

from .features import FourierFeatures, tetris_features, cartpole_features
from .sarsa_agent import SarsaLinearAgent
from .dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Semillas
# ---------------------------------------------------------------------------

def set_global_seed(seed: int):
    """Fija semillas en Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Preprocesamiento de observaciones
# ---------------------------------------------------------------------------

def cartpole_obs(obs) -> np.ndarray:
    return np.asarray(obs, dtype=np.float32)


def tetris_obs_flat(obs) -> np.ndarray:
    """Aplana la observación de Tetris para usarla como entrada a una red."""
    if isinstance(obs, dict):
        board = obs.get("board", list(obs.values())[0])
    else:
        board = obs
    return np.asarray(board, dtype=np.float32).flatten()


# ---------------------------------------------------------------------------
# Entrenamiento SARSA
# ---------------------------------------------------------------------------

def train_sarsa_cartpole(
    n_episodes:    int   = 600,
    seed:          int   = 0,
    alpha:         float = 3e-3,
    gamma:         float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end:   float = 0.01,
    epsilon_decay: float = 0.995,
    fourier_order: int   = 2,
) -> List[float]:
    """
    Entrena SARSA semi-gradiente en CartPole-v1.
    Devuelve lista de retornos por episodio.
    """
    set_global_seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    feat_fn = cartpole_features(fourier_order)
    agent   = SarsaLinearAgent(
        n_features    = feat_fn.n_features,
        n_actions     = env.action_space.n,
        alpha         = alpha,
        gamma         = gamma,
        epsilon_start = epsilon_start,
        epsilon_end   = epsilon_end,
        epsilon_decay = epsilon_decay,
        alpha_scales  = feat_fn.alpha_scaling(),
    )

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        phi    = feat_fn(cartpole_obs(obs))
        action = agent.select_action(phi)
        ep_ret = 0.0

        while True:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            phi_next   = feat_fn(cartpole_obs(next_obs))
            next_action = agent.select_action(phi_next)

            agent.update(phi, action, reward, phi_next, next_action, done)
            ep_ret += reward
            phi, action = phi_next, next_action

            if done:
                break

        agent.decay_epsilon()
        returns.append(ep_ret)

    env.close()
    return returns


def train_sarsa_tetris(
    n_episodes:    int   = 200,
    seed:          int   = 0,
    alpha:         float = 1e-2,
    gamma:         float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end:   float = 0.05,
    epsilon_decay: float = 0.99,
    max_steps:     int   = 2000,
) -> List[float]:
    """
    Entrena SARSA semi-gradiente en Tetris (tetris_gymnasium/Tetris).
    Devuelve lista de retornos por episodio.
    """
    set_global_seed(seed)
    from tetris_gymnasium.envs import Tetris as _TetrisEnv  # registra el env
    env = gym.make("tetris_gymnasium/Tetris")
    env.reset(seed=seed)

    from .features import N_TETRIS_FEATURES
    n_actions = env.action_space.n
    agent = SarsaLinearAgent(
        n_features    = N_TETRIS_FEATURES,
        n_actions     = n_actions,
        alpha         = alpha,
        gamma         = gamma,
        epsilon_start = epsilon_start,
        epsilon_end   = epsilon_end,
        epsilon_decay = epsilon_decay,
    )

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        phi    = tetris_features(obs)
        action = agent.select_action(phi)
        ep_ret = 0.0
        steps  = 0

        while steps < max_steps:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done        = terminated or truncated
            phi_next    = tetris_features(next_obs)
            next_action = agent.select_action(phi_next)

            agent.update(phi, action, reward, phi_next, next_action, done)
            ep_ret += reward
            phi, action = phi_next, next_action
            steps += 1

            if done:
                break

        agent.decay_epsilon()
        returns.append(ep_ret)

    env.close()
    return returns


# ---------------------------------------------------------------------------
# Entrenamiento DQN
# ---------------------------------------------------------------------------

def train_dqn_cartpole(
    n_episodes:         int   = 600,
    seed:               int   = 0,
    lr:                 float = 5e-4,
    gamma:              float = 0.99,
    epsilon_start:      float = 1.0,
    epsilon_end:        float = 0.01,
    epsilon_decay:      float = 0.9995,
    buffer_size:        int   = 20_000,
    batch_size:         int   = 64,
    target_update_freq: int   = 200,
    hidden_size:        int   = 128,
    n_layers:           int   = 2,
    learn_every:        int   = 1,
) -> List[float]:
    """
    Entrena DQN en CartPole-v1. Devuelve retornos por episodio.
    """
    set_global_seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    input_dim = env.observation_space.shape[0]
    agent = DQNAgent(
        input_dim          = input_dim,
        n_actions          = env.action_space.n,
        hidden_size        = hidden_size,
        n_layers           = n_layers,
        lr                 = lr,
        gamma              = gamma,
        epsilon_start      = epsilon_start,
        epsilon_end        = epsilon_end,
        epsilon_decay      = epsilon_decay,
        buffer_size        = buffer_size,
        batch_size         = batch_size,
        target_update_freq = target_update_freq,
    )

    returns = []
    step_count = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state  = cartpole_obs(obs)
        ep_ret = 0.0

        while True:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = cartpole_obs(next_obs)

            agent.store(state, action, reward, next_state, done)
            step_count += 1
            if step_count % learn_every == 0:
                agent.learn()

            agent.decay_epsilon()
            ep_ret += reward
            state   = next_state

            if done:
                break

        returns.append(ep_ret)

    env.close()
    return returns


def train_dqn_tetris(
    n_episodes:         int   = 200,
    seed:               int   = 0,
    lr:                 float = 1e-3,
    gamma:              float = 0.99,
    epsilon_start:      float = 1.0,
    epsilon_end:        float = 0.05,
    epsilon_decay:      float = 0.999,
    buffer_size:        int   = 20_000,
    batch_size:         int   = 64,
    target_update_freq: int   = 300,
    hidden_size:        int   = 256,
    n_layers:           int   = 2,
    max_steps:          int   = 2000,
) -> List[float]:
    """
    Entrena DQN en Tetris (tetris_gymnasium/Tetris).
    Devuelve retornos por episodio.
    """
    set_global_seed(seed)
    from tetris_gymnasium.envs import Tetris as _TetrisEnv  # registra el env
    env = gym.make("tetris_gymnasium/Tetris")
    env.reset(seed=seed)

    obs_sample, _ = env.reset(seed=seed)
    input_dim = len(tetris_obs_flat(obs_sample))

    agent = DQNAgent(
        input_dim          = input_dim,
        n_actions          = env.action_space.n,
        hidden_size        = hidden_size,
        n_layers           = n_layers,
        lr                 = lr,
        gamma              = gamma,
        epsilon_start      = epsilon_start,
        epsilon_end        = epsilon_end,
        epsilon_decay      = epsilon_decay,
        buffer_size        = buffer_size,
        batch_size         = batch_size,
        target_update_freq = target_update_freq,
    )

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        state  = tetris_obs_flat(obs)
        ep_ret = 0.0
        steps  = 0

        while steps < max_steps:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = tetris_obs_flat(next_obs)

            agent.store(state, action, reward, next_state, done)
            agent.learn()
            agent.decay_epsilon()

            ep_ret += reward
            state   = next_state
            steps  += 1

            if done:
                break

        returns.append(ep_ret)

    env.close()
    return returns


# ---------------------------------------------------------------------------
# Experimentos multi-semilla
# ---------------------------------------------------------------------------

def run_multiseed(
    train_fn:  Callable,
    seeds:     List[int],
    **kwargs,
) -> np.ndarray:
    """
    Ejecuta train_fn con cada semilla de `seeds` y devuelve una matriz
    (n_seeds × n_episodes) con los retornos.
    """
    all_returns = []
    for seed in seeds:
        rets = train_fn(seed=seed, **kwargs)
        all_returns.append(rets)
    return np.array(all_returns)               # (n_seeds, n_episodes)


# ---------------------------------------------------------------------------
# Métricas de resumen
# ---------------------------------------------------------------------------

def summary_stats(returns_matrix: np.ndarray, window: int = 50) -> dict:
    """
    Devuelve métricas de resumen para una matriz (n_seeds, n_episodes).

    Returns:
        dict con:
          - 'mean_curve'   : media suavizada (n_episodes,)
          - 'std_curve'    : desv. estándar suavizada (n_episodes,)
          - 'final_mean'   : media del retorno en los últimos `window` episodios
          - 'final_std'    : desv. estándar del retorno final
          - 'best_episode' : episodio con mayor retorno medio
    """
    def smooth(arr, w=window):
        return np.convolve(arr, np.ones(w) / w, mode='valid')

    smoothed = np.array([smooth(r) for r in returns_matrix])  # (n_seeds, T')
    mean_c   = smoothed.mean(axis=0)
    std_c    = smoothed.std(axis=0)

    final = returns_matrix[:, -window:]
    return {
        "mean_curve":   mean_c,
        "std_curve":    std_c,
        "final_mean":   float(final.mean()),
        "final_std":    float(final.std()),
        "best_episode": int(np.argmax(mean_c)),
    }
