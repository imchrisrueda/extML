"""
training.py
===========
Bucles de entrenamiento y evaluacion para los agentes SARSA y DQN.
Incluye semillas, episode_seed y funciones de evaluacion greedy (epsilon=0).

Nota metodologica sobre epsilon:
  Ambos algoritmos usan decaimiento por episodio, calibrado al presupuesto total
  (eps_decay = eps_end ^ (1 / n_episodes)).  Esto garantiza que SARSA y DQN
  consuman el mismo arco de exploracion y que la diferencia de resultados sea
  atribuible al algoritmo, no al horario de epsilon.
"""

import os
import random
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import gymnasium as gym

from .features import FourierFeatures, cartpole_features, tetris_features, get_board
from .sarsa_agent import SarsaLinearAgent
from .dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Semillas y reproducibilidad
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Fija semillas en Python, NumPy y PyTorch para reproducibilidad total."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def episode_seed(base_seed: int, episode_idx: int) -> int:
    """Semilla determinista por episodio: garantiza que SARSA y DQN vean la misma dinamica del entorno."""
    return int(base_seed * 100_000 + episode_idx)


# ---------------------------------------------------------------------------
# Entrenamiento SARSA
# ---------------------------------------------------------------------------

def train_sarsa_cartpole(
    seed:          int            = 0,
    n_episodes:    int            = 500,
    alpha:         float          = 3e-3,
    gamma:         float          = 0.99,
    eps_start:     float          = 1.0,
    eps_end:       float          = 0.01,
    eps_decay:     Optional[float] = None,
    fourier_order: int            = 2,
    max_steps:     int            = 500,
    return_agent:  bool           = False,
) -> Union[List[float], Tuple[SarsaLinearAgent, List[float]]]:
    """
    Entrena SARSA semi-gradiente con features de Fourier en CartPole-v1.

    eps_decay se calibra automaticamente a eps_end^(1/n_episodes) si no se
    proporciona, de modo que el agente llega exactamente a eps_end al final
    del entrenamiento.

    Args:
        seed:          Semilla base para el entrenamiento.
        n_episodes:    Numero de episodios de entrenamiento.
        alpha:         Tasa de aprendizaje.
        gamma:         Factor de descuento.
        eps_start:     Epsilon inicial.
        eps_end:       Epsilon final.
        eps_decay:     Multiplicador por episodio; None = calibracion automatica.
        fourier_order: Orden de la aproximacion de Fourier (k en k^n features).
        max_steps:     Pasos maximos por episodio.
        return_agent:  Si True devuelve (agente, retornos); si False solo retornos.

    Returns:
        Lista de retornos por episodio, o (SarsaLinearAgent, lista) si return_agent.
    """
    if eps_decay is None:
        eps_decay = float(eps_end ** (1.0 / n_episodes))

    set_global_seed(seed)
    env = gym.make("CartPole-v1")

    feat = cartpole_features(fourier_order)
    agent = SarsaLinearAgent(
        n_features    = feat.n_features,
        n_actions     = env.action_space.n,
        alpha         = alpha,
        gamma         = gamma,
        epsilon_start = eps_start,
        epsilon_end   = eps_end,
        epsilon_decay = eps_decay,
        alpha_scales  = feat.alpha_scaling(),
    )

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        phi    = feat(obs)
        action = agent.select_action(phi)
        ep_ret = 0.0

        for _ in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done        = terminated or truncated
            phi_next    = feat(next_obs)
            next_action = agent.select_action(phi_next)

            agent.update(phi, action, reward, phi_next, next_action, done)
            ep_ret += reward
            phi, action = phi_next, next_action

            if done:
                break

        agent.decay_epsilon()   # decaimiento por episodio
        returns.append(ep_ret)

    env.close()
    return (agent, returns) if return_agent else returns


def train_sarsa_tetris(
    seed:         int            = 0,
    n_episodes:   int            = 250,
    alpha:        float          = 1e-2,
    gamma:        float          = 0.99,
    eps_start:    float          = 1.0,
    eps_end:      float          = 0.05,
    eps_decay:    Optional[float] = None,
    max_steps:    int            = 2000,
    return_agent: bool           = False,
) -> Union[List[float], Tuple[SarsaLinearAgent, List[float]]]:
    """
    Entrena SARSA semi-gradiente con features artesanales en Tetris.

    eps_decay se calibra automaticamente si no se proporciona.
    """
    from .features import N_TETRIS_FEATURES

    if eps_decay is None:
        eps_decay = float(eps_end ** (1.0 / n_episodes))

    set_global_seed(seed)
    env = gym.make("tetris_gymnasium/Tetris")

    agent = SarsaLinearAgent(
        n_features    = N_TETRIS_FEATURES,
        n_actions     = env.action_space.n,
        alpha         = alpha,
        gamma         = gamma,
        epsilon_start = eps_start,
        epsilon_end   = eps_end,
        epsilon_decay = eps_decay,
    )

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        phi    = tetris_features(obs)
        action = agent.select_action(phi)
        ep_ret = 0.0

        for _ in range(max_steps):
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done        = terminated or truncated
            phi_next    = tetris_features(next_obs)
            next_action = agent.select_action(phi_next)

            agent.update(phi, action, reward, phi_next, next_action, done)
            ep_ret += reward
            phi, action = phi_next, next_action

            if done:
                break

        agent.decay_epsilon()   # decaimiento por episodio
        returns.append(ep_ret)

    env.close()
    return (agent, returns) if return_agent else returns


# ---------------------------------------------------------------------------
# Entrenamiento DQN
# ---------------------------------------------------------------------------

def train_dqn_cartpole(
    seed:         int            = 0,
    n_episodes:   int            = 500,
    lr:           float          = 5e-4,
    gamma:        float          = 0.99,
    eps_start:    float          = 1.0,
    eps_end:      float          = 0.01,
    eps_decay:    Optional[float] = None,
    buf_size:     int            = 20_000,
    batch_size:   int            = 64,
    target_freq:  int            = 200,
    hidden:       int            = 128,
    n_layers:     int            = 2,
    max_steps:    int            = 500,
    return_agent: bool           = False,
) -> Union[List[float], Tuple[DQNAgent, List[float]]]:
    """
    Entrena DQN en CartPole-v1 con estado crudo (4 dimensiones).

    El epsilon decae **por episodio** (igual que SARSA), calibrado a
    eps_end^(1/n_episodes) si no se proporciona eps_decay.
    Esto garantiza arcos de exploracion comparables entre los dos algoritmos.

    Args:
        seed:         Semilla base.
        n_episodes:   Numero de episodios.
        lr:           Tasa de aprendizaje del optimizador Adam.
        gamma:        Factor de descuento.
        eps_start:    Epsilon inicial.
        eps_end:      Epsilon final.
        eps_decay:    Multiplicador por episodio; None = calibracion automatica.
        buf_size:     Capacidad del replay buffer.
        batch_size:   Tamano del minibatch para cada actualizacion.
        target_freq:  Pasos entre copias de la red principal a la red objetivo.
        hidden:       Neuronas por capa oculta.
        n_layers:     Numero de capas ocultas.
        max_steps:    Pasos maximos por episodio.
        return_agent: Si True devuelve (agente, retornos).
    """
    if eps_decay is None:
        eps_decay = float(eps_end ** (1.0 / n_episodes))

    set_global_seed(seed)
    env = gym.make("CartPole-v1")

    agent = DQNAgent(
        input_dim          = env.observation_space.shape[0],
        n_actions          = env.action_space.n,
        hidden_size        = hidden,
        n_layers           = n_layers,
        lr                 = lr,
        gamma              = gamma,
        epsilon_start      = eps_start,
        epsilon_end        = eps_end,
        epsilon_decay      = eps_decay,
        buffer_size        = buf_size,
        batch_size         = batch_size,
        target_update_freq = target_freq,
    )

    returns    = []
    step_count = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        state  = np.asarray(obs, dtype=np.float32)
        ep_ret = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = np.asarray(next_obs, dtype=np.float32)

            agent.store(state, action, reward, next_state, done)
            step_count += 1
            agent.learn()   # no-op hasta que el buffer tenga suficientes muestras

            ep_ret += reward
            state   = next_state

            if done:
                break

        agent.decay_epsilon()   # decaimiento por episodio (simetrico con SARSA)
        returns.append(ep_ret)

    env.close()
    return (agent, returns) if return_agent else returns


def train_dqn_tetris(
    seed:         int            = 0,
    n_episodes:   int            = 250,
    lr:           float          = 1e-3,
    gamma:        float          = 0.99,
    eps_start:    float          = 1.0,
    eps_end:      float          = 0.05,
    eps_decay:    Optional[float] = None,
    buf_size:     int            = 20_000,
    batch_size:   int            = 64,
    target_freq:  int            = 300,
    hidden:       int            = 256,
    n_layers:     int            = 2,
    max_steps:    int            = 2000,
    return_agent: bool           = False,
) -> Union[List[float], Tuple[DQNAgent, List[float]]]:
    """
    Entrena DQN en Tetris (tetris_gymnasium/Tetris) con tablero aplanado.

    El epsilon decae **por episodio** (igual que SARSA y que train_dqn_cartpole).
    """
    if eps_decay is None:
        eps_decay = float(eps_end ** (1.0 / n_episodes))

    set_global_seed(seed)
    env = gym.make("tetris_gymnasium/Tetris")

    # Determina la dimension de entrada aplanando una observacion de muestra
    obs_sample, _ = env.reset(seed=seed)
    board = get_board(obs_sample)
    input_dim = int(np.prod(board.shape))

    agent = DQNAgent(
        input_dim          = input_dim,
        n_actions          = env.action_space.n,
        hidden_size        = hidden,
        n_layers           = n_layers,
        lr                 = lr,
        gamma              = gamma,
        epsilon_start      = eps_start,
        epsilon_end        = eps_end,
        epsilon_decay      = eps_decay,
        buffer_size        = buf_size,
        batch_size         = batch_size,
        target_update_freq = target_freq,
    )

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        state  = get_board(obs).flatten()
        ep_ret = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated
            next_state = get_board(next_obs).flatten()

            agent.store(state, action, reward, next_state, done)
            agent.learn()

            ep_ret += reward
            state   = next_state

            if done:
                break

        agent.decay_epsilon()   # decaimiento por episodio
        returns.append(ep_ret)

    env.close()
    return (agent, returns) if return_agent else returns


# ---------------------------------------------------------------------------
# Evaluacion greedy (epsilon = 0) sobre bateria de semillas fijas
# ---------------------------------------------------------------------------

def evaluate_sarsa_cartpole(
    agent:      SarsaLinearAgent,
    feat:       FourierFeatures,
    eval_seeds: List[int],
    max_steps:  int = 500,
) -> np.ndarray:
    """
    Ejecuta el agente SARSA con politica greedy (epsilon=0) en CartPole.

    Args:
        agent:      Agente entrenado.
        feat:       Transformacion de features de Fourier usada en el entrenamiento.
        eval_seeds: Lista de semillas para reseteados deterministicos.
        max_steps:  Limite de pasos por episodio.

    Returns:
        Array (len(eval_seeds),) con el retorno de cada episodio de evaluacion.
    """
    env     = gym.make("CartPole-v1")
    saved   = agent.epsilon
    agent.epsilon = 0.0   # politica greedy pura

    rewards = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        phi    = feat(obs)
        ep_ret = 0.0

        for _ in range(max_steps):
            action = agent.select_action(phi)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            phi     = feat(next_obs)
            ep_ret += reward
            if terminated or truncated:
                break

        rewards.append(ep_ret)

    agent.epsilon = saved
    env.close()
    return np.array(rewards, dtype=np.float32)


def evaluate_sarsa_tetris(
    agent:      SarsaLinearAgent,
    eval_seeds: List[int],
    max_steps:  int = 2000,
) -> np.ndarray:
    """
    Ejecuta el agente SARSA con politica greedy (epsilon=0) en Tetris.

    Returns:
        Array (len(eval_seeds),) con el retorno de cada episodio de evaluacion.
    """
    env     = gym.make("tetris_gymnasium/Tetris")
    saved   = agent.epsilon
    agent.epsilon = 0.0

    rewards = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        phi    = tetris_features(obs)
        ep_ret = 0.0

        for _ in range(max_steps):
            action = agent.select_action(phi)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            phi     = tetris_features(next_obs)
            ep_ret += reward
            if terminated or truncated:
                break

        rewards.append(ep_ret)

    agent.epsilon = saved
    env.close()
    return np.array(rewards, dtype=np.float32)


def evaluate_dqn_cartpole(
    agent:      DQNAgent,
    eval_seeds: List[int],
    max_steps:  int = 500,
) -> np.ndarray:
    """
    Ejecuta el agente DQN con politica greedy (agent.greedy_action) en CartPole.

    Returns:
        Array (len(eval_seeds),) con el retorno de cada episodio de evaluacion.
    """
    env = gym.make("CartPole-v1")

    rewards = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        state  = np.asarray(obs, dtype=np.float32)
        ep_ret = 0.0

        for _ in range(max_steps):
            action = agent.greedy_action(state)   # sin epsilon
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state   = np.asarray(next_obs, dtype=np.float32)
            ep_ret += reward
            if terminated or truncated:
                break

        rewards.append(ep_ret)

    env.close()
    return np.array(rewards, dtype=np.float32)


def evaluate_dqn_tetris(
    agent:      DQNAgent,
    eval_seeds: List[int],
    max_steps:  int = 2000,
) -> np.ndarray:
    """
    Ejecuta el agente DQN con politica greedy (agent.greedy_action) en Tetris.

    Returns:
        Array (len(eval_seeds),) con el retorno de cada episodio de evaluacion.
    """
    env = gym.make("tetris_gymnasium/Tetris")

    rewards = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=s)
        state  = get_board(obs).flatten()
        ep_ret = 0.0

        for _ in range(max_steps):
            action = agent.greedy_action(state)   # sin epsilon
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state   = get_board(next_obs).flatten()
            ep_ret += reward
            if terminated or truncated:
                break

        rewards.append(ep_ret)

    env.close()
    return np.array(rewards, dtype=np.float32)
