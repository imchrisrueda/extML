"""
sarsa_agent.py
==============
SARSA Semi-Gradiente con aproximación lineal de la función de valor Q.

Regla de actualización (Sutton & Barto §10.1):

    w ← w + α [r + γ Q(s',a';w) − Q(s,a;w)] ∇_w Q(s,a;w)

con Q(s,a;w) = w_a · φ(s), donde φ(s) es el vector de características del
estado y w_a es el vector de pesos asociado a la acción a.

La política es ε-greedy con decaimiento multiplicativo.
"""

import numpy as np


class SarsaLinearAgent:
    """
    Agente SARSA semi-gradiente con aproximación lineal Q(s,a;w) = w_a · φ(s).

    Parameters
    ----------
    n_features : int
        Dimensión del vector de características φ(s).
    n_actions : int
        Número de acciones discretas.
    alpha : float
        Tasa de aprendizaje base.
    gamma : float
        Factor de descuento.
    epsilon_start : float
        Valor inicial de ε.
    epsilon_end : float
        Valor mínimo de ε.
    epsilon_decay : float
        Factor multiplicativo de decaimiento por episodio.
    alpha_scales : ndarray | None
        Escalas individuales por feature (e.g., de Fourier).  Si se pasan,
        alpha efectivo para el feature i será alpha * alpha_scales[i].
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        alpha: float = 5e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        alpha_scales: np.ndarray | None = None,
    ):
        self.n_features   = n_features
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay

        # w tiene shape (n_actions, n_features)
        self.weights = np.zeros((n_actions, n_features), dtype=np.float64)

        # α efectivo por feature (broadcast seguro)
        if alpha_scales is not None:
            self.alpha = alpha * alpha_scales.astype(np.float64)   # (n_features,)
        else:
            self.alpha = alpha

    # ------------------------------------------------------------------
    # Función de valor Q
    # ------------------------------------------------------------------

    def q_values(self, phi: np.ndarray) -> np.ndarray:
        """Q(s,·;w) para todas las acciones. Devuelve array (n_actions,)."""
        return self.weights @ phi                     # (n_actions, n_feat) @ (n_feat,)

    def q(self, phi: np.ndarray, action: int) -> float:
        """Q(s,a;w) escalar."""
        return float(self.weights[action] @ phi)

    # ------------------------------------------------------------------
    # Política ε-greedy
    # ------------------------------------------------------------------

    def select_action(self, phi: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.q_values(phi)))

    # ------------------------------------------------------------------
    # Actualización SARSA semi-gradiente
    # ------------------------------------------------------------------

    def update(
        self,
        phi: np.ndarray,
        action: int,
        reward: float,
        phi_next: np.ndarray,
        next_action: int,
        done: bool,
    ) -> float:
        """Aplica la regla de actualización SARSA y devuelve el error TD."""
        q_cur  = self.q(phi, action)
        q_next = 0.0 if done else self.q(phi_next, next_action)
        td_err = reward + self.gamma * q_next - q_cur
        # Gradiente de Q(s,a;w) respecto a w_a es φ(s)
        self.weights[action] += self.alpha * td_err * phi
        return td_err

    # ------------------------------------------------------------------
    # Gestión de ε
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Decae ε un paso multiplicativo."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Serialización ligera (para guardar/comparar pesos)
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        return self.weights.copy()

    def set_weights(self, w: np.ndarray):
        self.weights = w.copy()
