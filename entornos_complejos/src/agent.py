"""
Archivo: agent.py

El objetivo de este archivo es definir la clase agente para usar en el apartado A de la parte "entornos complejos"
Según la documentación de Gymnasium, para construir un agente hay que crear una clase con el nombre de ese agente 
y añadirle los métodos get_action() y update()

"""

import numpy as np
import gymnasium as gym
from typing import Any

class AgentMC:
    def __init__(self, env:gym.Env, epsilon: float = 0.1, gamma: float = 1.0, decay: bool = False):
        """Inicializa todo lo necesario para el aprendizaje con Monte Carlo
        
        :param env: entorno
        :param epsilon: exploración
        :param gamma: factor de descuento
        :param decay: si epsilon decrece
        """

        self.env = env
        self.nA = env.action_space.n # Número de acciones
        self.nS = env.observation_space.n # Número de estados

        # Inicializar todo
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay = decay

        # Q(s,a)
        self.Q = np.zeros((self.nS, self.nA)) 

        # Contador de visitas
        self.returns_count = np.zeros((self.nS, self.nA)) 

        # Memoria del episodio
        self.episode = []

        # Estadísticas
        self.episode_return = 0.0
        self.episode_length = 0


    def get_action(self, state) -> Any:
        """
        Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente --> política epsilon-greedy basada en Q
        """
        probs = np.ones(self.nA) * self.epsilon / self.nA # Al principio todas las acciones tienen la misma probabilidad
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1.0 - self.epsilon)

        return np.random.choice(np.arange(self.nA), p=probs)


    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Con la muestra (s, a, s', r) e información complementaria aplicamos el algoritmo.
        Implementación Monte Carlo all-sisit
        """

        # Guardamos transición (solo s,a,r)
        self.episode.append((obs, action, reward))

        self.episode_return += reward
        self.episode_length += 1

        done = terminated or truncated

        if done:
            G = 0

            # Recorremos el episodio hacia atrás
            for t in reversed(range(len(self.episode))):
                s, a, r = self.episode[t]
                G = self.gamma * G + r

                self.returns_count[s, a] += 1
                alpha = 1.0 / self.returns_count[s, a]

                self.Q[s, a] += alpha * (G - self.Q[s, a])

            # Reset episodio
            self.episode = []
            self.episode_return = 0.0
            self.episode_length = 0


        def decay_epsilon(self, episode_number):
            if self.decay:
                self.epsilon = min(1.0, 1000.0 / (episode_number + 1))

