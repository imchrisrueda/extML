"""
Utilidades comunes para el notebook B_control_aprox.

Este módulo extrae la lógica reutilizable del notebook sin alterar el
protocolo experimental:
- seeds y construcción de entornos
- features compartidas
- agentes SARSA y DQN
- bucles de entrenamiento y evaluación
- barridos por semillas
- utilidades de resumen, visualización y simulación
"""

from __future__ import annotations

import os
import random
from itertools import product
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


CARTPOLE_LOW = np.array([-4.8, -3.5, -0.42, -3.5], dtype=np.float64)
CARTPOLE_HIGH = np.array([4.8, 3.5, 0.42, 3.5], dtype=np.float64)
N_TETRIS_FEAT = 7
_TETRIS_ENV_CLASS = None


def set_global_seed(seed: int) -> None:
    """Semillas en Python/NumPy/Torch + ajustes deterministas."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def episode_seed(base_seed: int, episode_idx: int) -> int:
    """Usa la misma semilla por episodio para comparar algoritmos bajo condiciones equivalentes."""
    return int(base_seed * 100_000 + episode_idx)


def ensure_tetris_env_imported():
    """Importa Tetris para forzar su registro en Gymnasium."""
    global _TETRIS_ENV_CLASS
    if _TETRIS_ENV_CLASS is None:
        from tetris_gymnasium.envs import Tetris as _TetrisEnvClass

        _TETRIS_ENV_CLASS = _TetrisEnvClass
    return _TETRIS_ENV_CLASS


def make_seeded_env(env_id: str, seed: int | None = None, render_mode: str | None = None, ensure_tetris: bool = False):
    """Construye un entorno, gestiona el render opcional y siembra sus espacios si procede."""
    if ensure_tetris:
        ensure_tetris_env_imported()

    make_kwargs = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    try:
        env = gym.make(env_id, **make_kwargs)
    except TypeError:
        if render_mode is None:
            raise
        env = gym.make(env_id)

    if seed is not None:
        env.action_space.seed(seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
    return env


class FourierFeatures:
    """Base de Fourier coseno para aproximación lineal en espacios continuos acotados."""

    def __init__(self, obs_low, obs_high, order: int = 2):
        self.low = np.asarray(obs_low, dtype=np.float64)
        self.rng = np.asarray(obs_high, dtype=np.float64) - self.low + 1e-8
        indices = list(product(range(order + 1), repeat=len(self.low)))
        self.C = np.array(indices, dtype=np.float64)
        self.n_features = len(self.C)

    def __call__(self, obs):
        """Mapea una observación continua al vector de características phi(s)."""
        s = np.clip(np.asarray(obs, dtype=np.float64), self.low, self.low + self.rng)
        s_norm = (s - self.low) / self.rng
        return np.cos(np.pi * (self.C @ s_norm))

    def alpha_scaling(self):
        """Escala alpha por frecuencia para estabilizar el aprendizaje con bases de Fourier."""
        norms = np.linalg.norm(self.C, axis=1)
        norms[norms == 0] = 1.0
        return 1.0 / norms


def cartpole_featurize(obs, feat: FourierFeatures) -> np.ndarray:
    """Convierte una observación de CartPole en features float32 reutilizables."""
    return feat(obs).astype(np.float32)


def _get_board(obs) -> np.ndarray:
    """Extrae el tablero 2D desde la observación del entorno Tetris."""
    if isinstance(obs, dict):
        board = obs.get("board", list(obs.values())[0])
    else:
        board = obs
    board = np.asarray(board, dtype=np.float32)
    if board.ndim > 2:
        board = board.squeeze()
    return board


def _column_heights(board: np.ndarray, occupied: np.ndarray | None = None) -> np.ndarray:
    """Calcula las alturas de columna de forma vectorizada."""
    if occupied is None:
        occupied = board > 0
    h_rows = board.shape[0]
    any_filled = occupied.any(axis=0)
    first_filled = np.argmax(occupied, axis=0)
    return np.where(any_filled, h_rows - first_filled, 0).astype(np.float32)


def tetris_featurize(obs) -> np.ndarray:
    """Construye un vector normalizado de 7 descriptores estructurales del tablero."""
    board = _get_board(obs)
    h_rows, w_cols = board.shape
    occupied = board > 0
    heights = _column_heights(board, occupied)

    agg_h = float(heights.sum())
    max_h = float(heights.max()) if w_cols > 0 else 0.0
    bumpy = float(np.abs(np.diff(heights)).sum())
    std_h = float(heights.std())
    complete = float(occupied.all(axis=1).sum())

    below_top = np.cumsum(occupied, axis=0) > 0
    holes = float(np.logical_and(~occupied, below_top).sum())

    if w_cols:
        left = np.empty_like(heights)
        right = np.empty_like(heights)
        left[0] = h_rows
        left[1:] = heights[:-1]
        right[-1] = h_rows
        right[:-1] = heights[1:]
        wells = float(np.clip(np.minimum(left, right) - heights, 0.0, None).sum())
    else:
        wells = 0.0

    divs = np.array([h_rows * w_cols, h_rows, h_rows * w_cols / 4.0, h_rows * w_cols, h_rows, 4.0, h_rows * w_cols], dtype=np.float32) + 1e-8
    raw = np.array([agg_h, max_h, holes, bumpy, std_h, complete, wells], dtype=np.float32)
    return np.clip(raw / divs, 0.0, 1.0)


class SarsaLinearAgent:
    """Agente SARSA semi-gradiente lineal."""

    def __init__(self, n_features, n_actions, alpha=5e-3, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995, alpha_scales=None):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.weights = np.zeros((n_actions, n_features), dtype=np.float64)
        self.alpha = alpha * alpha_scales.astype(np.float64) if alpha_scales is not None else float(alpha)

    def q_values(self, phi):
        """Evalúa Q(s, ·) para el vector de características actual."""
        return self.weights @ phi

    def select_action(self, phi):
        """Aplica epsilon-greedy sobre los valores-acción aproximados."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_values(phi)))

    def update(self, phi, action, reward, phi_next, next_action, done):
        """Ejecuta un paso de SARSA semi-gradiente lineal."""
        q_cur = float(self.weights[action] @ phi)
        q_next = 0.0 if done else float(self.weights[next_action] @ phi_next)
        delta = reward + self.gamma * q_next - q_cur
        self.weights[action] += self.alpha * delta * phi
        return delta

    def decay_epsilon(self):
        """Reduce epsilon por episodio hasta el mínimo fijado."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


class ReplayBuffer:
    """Buffer FIFO de transiciones para replay experience con muestreo uniforme sin reemplazo."""

    def __init__(self, capacity=20_000):
        self.capacity = int(capacity)
        self._size = 0
        self._pos = 0
        self._state_shape = None
        self._states = None
        self._next_states = None
        self._actions = np.empty(self.capacity, dtype=np.int64)
        self._rewards = np.empty(self.capacity, dtype=np.float32)
        self._dones = np.empty(self.capacity, dtype=np.float32)

    def _ensure_storage(self, state_shape):
        if self._states is not None:
            if tuple(state_shape) != self._state_shape:
                raise ValueError(f"ReplayBuffer recibió estados con shape inconsistente: {state_shape} != {self._state_shape}")
            return
        self._state_shape = tuple(state_shape)
        full_shape = (self.capacity, *self._state_shape)
        self._states = np.empty(full_shape, dtype=np.float32)
        self._next_states = np.empty(full_shape, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        """Almacena una transición ya convertida a dtypes compactos."""
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self._ensure_storage(state.shape)
        idx = self._pos
        self._states[idx] = state
        self._next_states[idx] = next_state
        self._actions[idx] = int(action)
        self._rewards[idx] = float(reward)
        self._dones[idx] = float(done)
        self._pos = (idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        """Extrae un minibatch uniforme."""
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            torch.from_numpy(self._states[idx]),
            torch.from_numpy(self._actions[idx]),
            torch.from_numpy(self._rewards[idx]),
            torch.from_numpy(self._next_states[idx]),
            torch.from_numpy(self._dones[idx]),
        )

    def __len__(self):
        return self._size


class QNetwork(nn.Module):
    """MLP totalmente conectada que aproxima Q(s, ·)."""

    def __init__(self, in_dim, out_dim, hidden=128, n_layers=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Propaga un batch y devuelve un valor-acción por acción disponible."""
        return self.net(x)


class DQNAgent:
    """Deep Q-Network con replay, target network y decaimiento epsilon por episodio."""

    def __init__(
        self,
        input_dim,
        n_actions,
        hidden=128,
        n_layers=2,
        lr=5e-4,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        buf_size=20_000,
        batch=64,
        target_freq=200,
        grad_clip=10.0,
        learning_starts=256,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch = batch
        self.target_freq = target_freq
        self.grad_clip = grad_clip
        self.learning_starts = max(int(learning_starts), int(batch))
        self._learn_steps = 0

        self.q_net = QNetwork(input_dim, n_actions, hidden, n_layers).to(self.device)
        self.target_net = QNetwork(input_dim, n_actions, hidden, n_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buf_size)

    def select_action(self, state):
        """Selecciona acción epsilon-greedy sobre la red online."""
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(1).item())

    def store(self, state, action, reward, next_state, done):
        """Inserta la transición observada en el replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        """Ejecuta una actualización DQN si ya hay suficientes datos en el buffer."""
        if len(self.buffer) < self.learning_starts:
            return None
        s, a, r, ns, d = [t.to(self.device) for t in self.buffer.sample(self.batch)]
        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1).values
            q_target = r + self.gamma * q_next * (1.0 - d)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

    def decay_epsilon(self):
        """Reduce epsilon una vez por episodio."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)


def sarsa_greedy_action(agent: SarsaLinearAgent, phi) -> int:
    """Devuelve la acción greedy de SARSA para un vector de características ya calculado."""
    return int(np.argmax(agent.q_values(phi)))


def dqn_greedy_action(agent: DQNAgent, state) -> int:
    """Devuelve la acción greedy de DQN para un estado ya codificado."""
    with torch.no_grad():
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
        return int(agent.q_net(st).argmax(1).item())


def _build_sarsa_cartpole_agent(env, alpha, gamma, eps_start, eps_end, eps_decay, fourier_order):
    feat = FourierFeatures(CARTPOLE_LOW, CARTPOLE_HIGH, order=fourier_order)
    agent = SarsaLinearAgent(
        feat.n_features,
        env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        alpha_scales=feat.alpha_scaling(),
    )
    return agent, feat


def _build_sarsa_tetris_agent(env, alpha, gamma, eps_start, eps_end, eps_decay):
    agent = SarsaLinearAgent(
        N_TETRIS_FEAT,
        env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
    )
    return agent, None


def _train_sarsa(seed, n_episodes, env_id, agent_builder, state_encoder, max_steps, return_agent=False):
    """Bucle común de entrenamiento SARSA para evitar duplicación entre entornos."""
    set_global_seed(seed)
    env = make_seeded_env(env_id, seed=seed, ensure_tetris=(env_id == "tetris_gymnasium/Tetris"))
    agent, state_aux = agent_builder(env)

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        state = state_encoder(obs, state_aux)
        action = agent.select_action(state)
        ep_ret, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_encoder(next_obs, state_aux)
            next_action = 0 if done else agent.select_action(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            ep_ret += reward
            state, action = next_state, next_action
            steps += 1
        agent.decay_epsilon()
        returns.append(ep_ret)

    env.close()
    if return_agent:
        result = (returns, agent)
        return result + ((state_aux,) if state_aux is not None else tuple())
    return returns


def _evaluate_sarsa(agent, env_id, state_encoder, eval_seeds, max_steps, state_aux=None):
    """Evaluación greedy común de SARSA con una batería fija de semillas."""
    env = make_seeded_env(env_id, ensure_tetris=(env_id == "tetris_gymnasium/Tetris"))
    rets = []
    for seed in eval_seeds:
        obs, _ = env.reset(seed=seed)
        state = state_encoder(obs, state_aux)
        done, ep_ret, steps = False, 0.0, 0
        while not done and steps < max_steps:
            action = sarsa_greedy_action(agent, state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            state = state_encoder(next_obs, state_aux)
            steps += 1
        rets.append(ep_ret)
    env.close()
    return np.array(rets, dtype=float)


def train_sarsa_cartpole(
    seed=0,
    n_episodes=500,
    alpha=3e-3,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    fourier_order=2,
    max_steps=500,
    return_agent=False,
):
    """Entrena SARSA semi-gradiente en CartPole con la base de Fourier compartida."""
    return _train_sarsa(
        seed=seed,
        n_episodes=n_episodes,
        env_id="CartPole-v1",
        agent_builder=lambda env: _build_sarsa_cartpole_agent(env, alpha, gamma, eps_start, eps_end, eps_decay, fourier_order),
        state_encoder=lambda obs, feat: feat(obs),
        max_steps=max_steps,
        return_agent=return_agent,
    )


def train_sarsa_tetris(
    seed=0,
    n_episodes=250,
    alpha=1e-2,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.99,
    max_steps=800,
    return_agent=False,
):
    """Entrena SARSA semi-gradiente en Tetris sobre la representación compacta compartida."""
    return _train_sarsa(
        seed=seed,
        n_episodes=n_episodes,
        env_id="tetris_gymnasium/Tetris",
        agent_builder=lambda env: _build_sarsa_tetris_agent(env, alpha, gamma, eps_start, eps_end, eps_decay),
        state_encoder=lambda obs, _: tetris_featurize(obs),
        max_steps=max_steps,
        return_agent=return_agent,
    )


def evaluate_sarsa_cartpole(agent, feat, eval_seeds, max_steps=500):
    """Evalúa la política greedy de SARSA en CartPole."""
    return _evaluate_sarsa(
        agent=agent,
        env_id="CartPole-v1",
        state_encoder=lambda obs, aux: aux(obs),
        eval_seeds=eval_seeds,
        max_steps=max_steps,
        state_aux=feat,
    )


def evaluate_sarsa_tetris(agent, eval_seeds, max_steps=800):
    """Evalúa la política greedy de SARSA en Tetris."""
    return _evaluate_sarsa(
        agent=agent,
        env_id="tetris_gymnasium/Tetris",
        state_encoder=lambda obs, _: tetris_featurize(obs),
        eval_seeds=eval_seeds,
        max_steps=max_steps,
    )


def _build_dqn_cartpole_agent(env, lr, gamma, eps_start, eps_end, eps_decay, buf_size, batch, target_freq, hidden, n_layers, learning_starts):
    feat = FourierFeatures(CARTPOLE_LOW, CARTPOLE_HIGH, order=2)
    agent = DQNAgent(
        input_dim=feat.n_features,
        n_actions=env.action_space.n,
        hidden=hidden,
        n_layers=n_layers,
        lr=lr,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        buf_size=buf_size,
        batch=batch,
        target_freq=target_freq,
        learning_starts=learning_starts,
    )
    return agent, feat


def _build_dqn_tetris_agent(env, lr, gamma, eps_start, eps_end, eps_decay, buf_size, batch, target_freq, hidden, n_layers, learning_starts):
    agent = DQNAgent(
        input_dim=N_TETRIS_FEAT,
        n_actions=env.action_space.n,
        hidden=hidden,
        n_layers=n_layers,
        lr=lr,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        buf_size=buf_size,
        batch=batch,
        target_freq=target_freq,
        learning_starts=learning_starts,
    )
    return agent, None


def _train_dqn(seed, n_episodes, env_id, agent_builder, state_encoder, max_steps, train_freq, return_agent=False):
    """Bucle común de entrenamiento DQN para evitar duplicación entre entornos."""
    set_global_seed(seed)
    env = make_seeded_env(env_id, seed=seed, ensure_tetris=(env_id == "tetris_gymnasium/Tetris"))
    agent, state_aux = agent_builder(env)

    returns = []
    env_steps = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=episode_seed(seed, ep))
        state = state_encoder(obs, state_aux)
        ep_ret, done, steps = 0.0, False, 0
        while not done and steps < max_steps:
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_encoder(next_obs, state_aux)
            agent.store(state, action, reward, next_state, done)
            env_steps += 1
            if env_steps % train_freq == 0:
                agent.learn()
            ep_ret += reward
            state = next_state
            steps += 1
        agent.decay_epsilon()
        returns.append(ep_ret)

    env.close()
    if return_agent:
        result = (returns, agent)
        return result + ((state_aux,) if state_aux is not None else tuple())
    return returns


def _evaluate_dqn(agent, env_id, state_encoder, eval_seeds, max_steps, state_aux=None):
    """Evaluación greedy común de DQN con una batería fija de semillas."""
    env = make_seeded_env(env_id, ensure_tetris=(env_id == "tetris_gymnasium/Tetris"))
    rets = []
    for seed in eval_seeds:
        obs, _ = env.reset(seed=seed)
        state = state_encoder(obs, state_aux)
        done, ep_ret, steps = False, 0.0, 0
        while not done and steps < max_steps:
            action = dqn_greedy_action(agent, state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward
            state = state_encoder(next_obs, state_aux)
            steps += 1
        rets.append(ep_ret)
    env.close()
    return np.array(rets, dtype=float)


def train_dqn_cartpole(
    seed=0,
    n_episodes=500,
    lr=5e-4,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995,
    buf_size=15_000,
    batch=64,
    target_freq=120,
    hidden=64,
    n_layers=2,
    learning_starts=256,
    train_freq=4,
    max_steps=500,
    return_agent=False,
):
    """Entrena DQN en CartPole sobre la misma base de Fourier que SARSA."""
    return _train_dqn(
        seed=seed,
        n_episodes=n_episodes,
        env_id="CartPole-v1",
        agent_builder=lambda env: _build_dqn_cartpole_agent(env, lr, gamma, eps_start, eps_end, eps_decay, buf_size, batch, target_freq, hidden, n_layers, learning_starts),
        state_encoder=lambda obs, feat: cartpole_featurize(obs, feat),
        max_steps=max_steps,
        train_freq=train_freq,
        return_agent=return_agent,
    )


def train_dqn_tetris(
    seed=0,
    n_episodes=250,
    lr=7e-4,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.99,
    buf_size=15_000,
    batch=64,
    target_freq=180,
    hidden=96,
    n_layers=2,
    learning_starts=512,
    train_freq=4,
    max_steps=800,
    return_agent=False,
):
    """Entrena DQN en Tetris usando los mismos 7 rasgos que SARSA."""
    return _train_dqn(
        seed=seed,
        n_episodes=n_episodes,
        env_id="tetris_gymnasium/Tetris",
        agent_builder=lambda env: _build_dqn_tetris_agent(env, lr, gamma, eps_start, eps_end, eps_decay, buf_size, batch, target_freq, hidden, n_layers, learning_starts),
        state_encoder=lambda obs, _: tetris_featurize(obs).astype(np.float32),
        max_steps=max_steps,
        train_freq=train_freq,
        return_agent=return_agent,
    )


def evaluate_dqn_cartpole(agent, feat, eval_seeds, max_steps=500):
    """Evalúa la política greedy de DQN en CartPole."""
    return _evaluate_dqn(
        agent=agent,
        env_id="CartPole-v1",
        state_encoder=lambda obs, aux: cartpole_featurize(obs, aux),
        eval_seeds=eval_seeds,
        max_steps=max_steps,
        state_aux=feat,
    )


def evaluate_dqn_tetris(agent, eval_seeds, max_steps=800):
    """Evalúa la política greedy de DQN en Tetris."""
    return _evaluate_dqn(
        agent=agent,
        env_id="tetris_gymnasium/Tetris",
        state_encoder=lambda obs, _: tetris_featurize(obs).astype(np.float32),
        eval_seeds=eval_seeds,
        max_steps=max_steps,
    )


def run_seed_sweep(run_label, seeds, train_fn, eval_fn, final_window, decimals=3, done_label="Listo."):
    """Ejecuta un barrido por semillas, evalúa la política final y conserva la primera como demo."""
    print(f"Entrenando {run_label}...")
    metric_fmt = f"{{:.{decimals}f}}"
    train_all, eval_all = [], []
    demo_payload = None

    for idx, seed in enumerate(seeds):
        print(f"  seed={seed}...", end=" ", flush=True)
        train_out = train_fn(seed)
        rets, *payload = train_out
        eval_rets = eval_fn(*payload)
        print(
            f"train-ult{final_window}=" + metric_fmt.format(np.mean(rets[-final_window:])) +
            " | eval=" + metric_fmt.format(np.mean(eval_rets))
        )
        train_all.append(rets)
        eval_all.append(eval_rets)
        if idx == 0:
            demo_payload = tuple(payload)

    print(done_label)
    return np.array(train_all), np.array(eval_all), demo_payload


def _moving_average_1d(arr, w):
    """Media móvil 1D vía suma acumulada para evitar trabajo repetido."""
    arr = np.asarray(arr, dtype=float)
    if arr.size < w:
        return arr.copy()
    csum = np.cumsum(np.pad(arr, (1, 0), mode="constant"))
    return (csum[w:] - csum[:-w]) / w


def _smoothed_matrix_and_x(returns_all, w):
    """Suaviza una matriz de retornos y devuelve también el eje x alineado."""
    returns_all = np.asarray(returns_all, dtype=float)
    if returns_all.ndim == 1:
        returns_all = returns_all[None, :]
    if returns_all.shape[1] < w:
        return returns_all.copy(), np.arange(returns_all.shape[1])
    csum = np.cumsum(np.pad(returns_all, ((0, 0), (1, 0)), mode="constant"), axis=1)
    smoothed = (csum[:, w:] - csum[:, :-w]) / w
    return smoothed, np.arange(w - 1, returns_all.shape[1])


def smooth(arr, w):
    """Media móvil en modo valid para visualizar tendencia sin alterar los datos crudos."""
    return _moving_average_1d(arr, w)


def smooth_x(raw_len, w):
    """Alinea el eje x con una media móvil calculada en modo valid."""
    if raw_len < w:
        return np.arange(raw_len)
    return np.arange(w - 1, raw_len)


def plot_band(ax, returns_all, color, label, w):
    """Dibuja la media suavizada y la banda de +-1 desviación típica entre semillas."""
    sm, x = _smoothed_matrix_and_x(returns_all, w)
    mean, std = sm.mean(0), sm.std(0)
    ax.plot(x, mean, color=color, label=label, linewidth=1.8)
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)


def final_window_seed_means(train_arr, w):
    """Promedia la ventana final de cada semilla para resumir el régimen terminal."""
    return np.asarray(train_arr[:, -w:].mean(axis=1), dtype=float)


def initial_window_seed_means(train_arr, w):
    """Promedia la ventana inicial de cada semilla para cuantificar la ganancia neta."""
    return np.asarray(train_arr[:, :w].mean(axis=1), dtype=float)


def seed_mean_stats(values):
    """Devuelve media, desviación típica poblacional y vector original."""
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std()), values


def eval_seed_means(eval_arr):
    """Resume cada semilla de evaluación greedy por la media de sus episodios evaluados."""
    return np.asarray(eval_arr.mean(axis=1), dtype=float)


def best_smoothed_episode(train_arr, w):
    """Localiza el episodio asociado al máximo de la curva media suavizada."""
    sm, x = _smoothed_matrix_and_x(train_arr, w)
    return int(x[int(sm.mean(0).argmax())])


def first_smoothed_crossing(train_arr, threshold, w):
    """Primer episodio en que la media suavizada supera un umbral descriptivo."""
    sm, x = _smoothed_matrix_and_x(train_arr, w)
    mean_curve = sm.mean(0)
    idx = np.where(mean_curve >= threshold)[0]
    return int(x[idx[0]]) if len(idx) else None


def fmt_episode(ep):
    """Formatea un episodio o informa de que el hito no se alcanzó."""
    return f"episodio {ep:,}" if ep is not None else "no alcanzado en el experimento"


def describe_gap(delta, tol=1e-12):
    """Describe únicamente el signo de una diferencia."""
    if delta > tol:
        return "supera"
    if delta < -tol:
        return "queda por debajo de"
    return "coincide con"


def compare_at_precision(a, b, decimals):
    """Compara dos métricas a la misma precisión con la que se reportan."""
    fa = float(f"{float(a):.{decimals}f}")
    fb = float(f"{float(b):.{decimals}f}")
    return int(fa > fb) - int(fa < fb)


def report_train_eval_summary(train_arr, eval_arr, color, label, title, final_window, decimals, smooth_w, extra_hlines=None):
    """Dibuja el resumen train/eval de una configuración y reporta medias finales agregadas por semilla."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_band(ax, train_arr, color, label, smooth_w)
    for line in extra_hlines or []:
        ax.axhline(
            line["y"],
            color=line.get("color", "gray"),
            linestyle=line.get("linestyle", "--"),
            linewidth=line.get("linewidth", 0.9),
            label=line.get("label"),
        )
    ax.set_xlabel("Episodio")
    ax.set_ylabel(f"Retorno (MA-{smooth_w})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    train_seed = final_window_seed_means(train_arr, final_window)
    eval_seed = eval_seed_means(eval_arr)
    fmt = f"{{:.{decimals}f}}"
    print(
        f"Train media final (media de semillas, ult.{final_window} ep): "
        + fmt.format(train_seed.mean()) + " +/- " + fmt.format(train_seed.std())
    )
    print("Eval greedy final (media de semillas): " + fmt.format(eval_seed.mean()) + " +/- " + fmt.format(eval_seed.std()))


def plot_eval_bars(ax, eval_arr_a, eval_arr_b, labels=("SARSA", "DQN")):
    """Dibuja barras de evaluación final con error +-sigma entre semillas."""
    vals, errs = [], []
    for eval_arr in (eval_arr_a, eval_arr_b):
        if eval_arr is None:
            vals.append(np.nan)
            errs.append(0.0)
        else:
            means = eval_seed_means(eval_arr)
            vals.append(float(np.mean(means)))
            errs.append(float(np.std(means)))
    x = np.arange(len(labels))
    colors = ["#1f77b4", "#d62728"]
    ax.bar(x, vals, yerr=errs, capsize=6, color=colors, alpha=0.85)
    ax.set_xticks(x, labels)
    ax.grid(True, axis="y", alpha=0.3)


def animate_frames(frames, title="Simulación", fps=20, max_frames=250):
    """Renderiza una secuencia de frames en HTML para inspección cualitativa."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    from matplotlib import animation

    if frames is None or len(frames) == 0:
        print(f"{title}: no se pudieron obtener frames.")
        return
    frames = frames[:max_frames]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis("off")
    ax.set_title(title)
    im = ax.imshow(frames[0])

    def _update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=1000 / fps, blit=True)
    plt.close(fig)
    display(HTML(ani.to_jshtml()))


def _append_rendered_frame(env, frames, allow_missing_shape=False, ignore_errors=False):
    """Añade un frame renderizado a una lista si está disponible."""
    try:
        frame = env.render()
    except Exception:
        if ignore_errors:
            return
        raise
    if frame is None:
        return
    if allow_missing_shape or hasattr(frame, "shape"):
        frames.append(frame)


def _rollout_env(policy_fn, env_id, seed, max_steps, ensure_tetris=False, strict_frame_shape=False):
    """Bucle común de rollout para simulaciones cualitativas."""
    env = make_seeded_env(env_id, render_mode="rgb_array", ensure_tetris=ensure_tetris)
    obs, _ = env.reset(seed=seed)
    frames, ret = [], 0.0
    ignore_errors = strict_frame_shape
    try:
        _append_rendered_frame(env, frames, allow_missing_shape=not strict_frame_shape, ignore_errors=ignore_errors)
        done, steps = False, 0
        while not done and steps < max_steps:
            action = int(policy_fn(obs))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
            _append_rendered_frame(env, frames, allow_missing_shape=not strict_frame_shape, ignore_errors=ignore_errors)
            steps += 1
    finally:
        env.close()
    return frames, ret


def rollout_cartpole(policy_fn, seed=123, max_steps=500):
    """Ejecuta una política en CartPole y devuelve frames y retorno."""
    return _rollout_env(policy_fn, "CartPole-v1", seed, max_steps, ensure_tetris=False, strict_frame_shape=False)


def rollout_tetris(policy_fn, env_id="tetris_gymnasium/Tetris", seed=456, max_steps=800):
    """Ejecuta una política en Tetris y devuelve frames y retorno."""
    return _rollout_env(policy_fn, env_id, seed, max_steps, ensure_tetris=True, strict_frame_shape=True)


def _build_policy(agent, state_encoder, action_fn):
    """Construye una política greedy a partir de un codificador de estado."""
    return lambda obs: action_fn(agent, state_encoder(obs))


def build_sarsa_cartpole_policy(agent, feat):
    """Construye la política greedy de SARSA para CartPole."""
    return _build_policy(agent, feat, sarsa_greedy_action)


def build_dqn_cartpole_policy(agent, feat):
    """Construye la política greedy de DQN para CartPole."""
    return _build_policy(agent, lambda obs: cartpole_featurize(obs, feat), dqn_greedy_action)


def build_sarsa_tetris_policy(agent):
    """Construye la política greedy de SARSA para Tetris."""
    return _build_policy(agent, tetris_featurize, sarsa_greedy_action)


def build_dqn_tetris_policy(agent):
    """Construye la política greedy de DQN para Tetris."""
    return _build_policy(agent, lambda obs: tetris_featurize(obs).astype(np.float32), dqn_greedy_action)


def save_gif(frames, filename, gif_dir: Path | str, imageio_module, fps=12):
    """Serializa una lista de frames en un GIF."""
    if frames is None or len(frames) == 0:
        print(f"No se guardó {filename}: no hay frames disponibles.")
        return
    gif_dir = Path(gif_dir)
    out_path = gif_dir / filename
    duration = 1.0 / max(1, int(fps))
    imageio_module.mimsave(out_path, [np.asarray(f) for f in frames], format="GIF", duration=duration, loop=0)
    print(f"GIF guardado en: {out_path}")
