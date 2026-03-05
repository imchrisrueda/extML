"""
dqn_agent.py
============
Deep Q-Network (DQN) según Mnih et al. 2015, con las buenas prácticas del
enunciado:

  - Q-network  (online)
  - Target network  (copia congelada con actualización periódica)
  - Replay buffer  (uniform random sampling)
  - Pérdida Huber (smooth L1)
  - Gradient clipping  (max_norm = 10)
  - Política ε-greedy con decaimiento lineal o multiplicativo

La red puede usarse tanto para CartPole (entrada 4D) como para Tetris
(entrada = tablero aplanado), pasando el tamaño de entrada correcto.
"""

import random
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Memoria de experiencias con muestreo uniforme.

    Almacena transiciones (s, a, r, s', done) como arrays numpy para
    eficiencia al construir minibatches.
    """

    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        self.buf.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states,      dtype=np.float32)),
            torch.tensor(np.array(actions,     dtype=np.int64)),
            torch.tensor(np.array(rewards,     dtype=np.float32)),
            torch.tensor(np.array(next_states, dtype=np.float32)),
            torch.tensor(np.array(dones,       dtype=np.float32)),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """MLP fully-connected con activaciones ReLU.

    Arquitectura:  input → hidden × n_layers → output (acciones)
    """

    def __init__(
        self,
        input_dim:   int,
        output_dim:  int,
        hidden_size: int = 128,
        n_layers:    int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_size), nn.ReLU()]
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Agente DQN completo.

    Parameters
    ----------
    input_dim : int         Dimensión del estado aplanado.
    n_actions : int         Número de acciones.
    hidden_size : int       Neuronas por capa oculta.
    n_layers : int          Número de capas ocultas.
    lr : float              Tasa de aprendizaje (Adam).
    gamma : float           Factor de descuento.
    epsilon_start : float   ε inicial.
    epsilon_end : float     ε mínimo.
    epsilon_decay : float   Decaimiento multiplicativo por paso.
    buffer_size : int       Capacidad del replay buffer.
    batch_size : int        Tamaño del minibatch.
    target_update_freq : int   Pasos entre actualizaciones de la target network.
    grad_clip : float       max_norm para gradient clipping.
    device : str            'cpu' | 'cuda' | 'auto'
    """

    def __init__(
        self,
        input_dim:         int,
        n_actions:         int,
        hidden_size:       int   = 128,
        n_layers:          int   = 2,
        lr:                float = 1e-3,
        gamma:             float = 0.99,
        epsilon_start:     float = 1.0,
        epsilon_end:       float = 0.01,
        epsilon_decay:     float = 0.9995,
        buffer_size:       int   = 20_000,
        batch_size:        int   = 64,
        target_update_freq: int  = 200,
        grad_clip:         float = 10.0,
        device:            str   = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.n_actions  = n_actions
        self.gamma      = gamma
        self.epsilon    = epsilon_start
        self.eps_end    = epsilon_end
        self.eps_decay  = epsilon_decay
        self.batch_size = batch_size
        self.target_freq = target_update_freq
        self.grad_clip   = grad_clip
        self._steps      = 0

        # Redes
        self.q_net      = QNetwork(input_dim, n_actions, hidden_size, n_layers).to(self.device)
        self.target_net = QNetwork(input_dim, n_actions, hidden_size, n_layers).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn   = nn.SmoothL1Loss()      # Huber loss

        self.buffer = ReplayBuffer(buffer_size)

    # ------------------------------------------------------------------
    # Política ε-greedy
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return int(self.q_net(s).argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Almacenar transición
    # ------------------------------------------------------------------

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    # ------------------------------------------------------------------
    # Paso de aprendizaje
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        """Muestrea el buffer y actualiza la Q-network. Devuelve la loss."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = [
            t.to(self.device) for t in self.buffer.sample(self.batch_size)
        ]

        # Q(s,a) con la red online
        q_pred = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + γ max Q_target(s',a')  (0 si terminal)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1.0 - dones)

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self._steps += 1

        # Actualización periódica de la target network
        if self._steps % self.target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ------------------------------------------------------------------
    # Gestión de ε
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Decae ε un paso multiplicativo."""
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ------------------------------------------------------------------
    # Guardar / cargar pesos
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
