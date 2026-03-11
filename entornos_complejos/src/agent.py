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
        self.nA = env.action_space.n # num de acciones
        self.nS = env.observation_space.n # num de estados

        self.epsilon = epsilon
        self.gamma = gamma
        self.decay = decay

        # Q(s,a) - valor esperado 
        self.Q = np.zeros((self.nS, self.nA)) 
        self.returns_count = np.zeros((self.nS, self.nA)) # contador de visitas
        self.C = np.zeros((self.nS, self.nA)) # suma de pesos     

        self.episode = []
        self.episode_return = 0.0
        self.episode_length = 0


    def get_action(self, state) -> Any:
        """
        Indicará qué acción realizar de acuerdo al estado.
        Responde a la política del agente --> política epsilon-greedy basada en Q
        """
        probs = np.ones(self.nA) * self.epsilon / self.nA # Al principio todas las acciones tienen la misma probabilidad
        best_action = np.argmax(self.Q[state])  # La mejor acción es la de más Q
        probs[best_action] += (1.0 - self.epsilon) # Ahora la mayor prob va a tener la de mayor Q

        return np.random.choice(np.arange(self.nA), p=probs) # En base a las probabilidades elegir una acción


    def update(self, obs, action, next_obs, reward, terminated, truncated, info):
        """
        Con la muestra (s, a, s', r) e información complementaria aplicamos el algoritmo.
        Implementación Monte Carlo all-sisit
        """

        # Guardar combinación - s,a,r
        self.episode.append((obs, action, reward))

        self.episode_return += reward # cuánto retorno ha tenido ese episodio
        self.episode_length += 1 # cuánto dura el episodio

        done = terminated or truncated

        # Si el episodio ha terminado se actualiza MC
        if done:
            G = 0

            for t in reversed(range(len(self.episode))): # reversed porque es hacia atrás
                s, a, r = self.episode[t] # Sacar valores del episodio
                G = self.gamma * G + r # Retorno

                self.returns_count[s, a] += 1 
                alpha = 1.0 / self.returns_count[s, a] # promedio 

                self.Q[s, a] += alpha * (G - self.Q[s, a])

            # Reset episodio
            self.episode = []
            self.episode_return = 0.0
            self.episode_length = 0

    

    # para off policy
    def behavior_action(self, state):
        """
        Política de comportamiento b(a|s)
        Totalmente aleatoria 
        """
        probs = np.ones(self.nA) * self.epsilon / self.nA
        best_action = np.argmax(self.Q[state])
        probs[best_action] += (1.0 - self.epsilon)
        return np.random.choice(np.arange(self.nA), p=probs)
    
     # este caso de off policy 
    def update_off(self, obs, action, next_obs, reward, terminated, truncated, info):
        self.episode.append((obs, action, reward))
        self.episode_return += reward
        self.episode_length += 1

        done = terminated or truncated

        if done:
            G = 0.0 # retorno acumulado 
            W = 1.0  # ratio de importancia acumulado

            for t in reversed(range(len(self.episode))): # recorrer hacia atrás
                s, a, r = self.episode[t]
                G = r + self.gamma * G

                # Actualización con Weighted Importance Sampling
                self.C[s, a] += W
                self.returns_count[s, a] += 1
                self.Q[s, a] += (W / self.C[s, a]) * (G - self.Q[s, a])

                # Calcular ratio de importancia para el paso anterior
                # Política objetivo π: greedy respecto a Q
                greedy_action = np.argmax(self.Q[s])
                
                # π(a|s) = 1 si a es la acción greedy, 0 si no
                if a != greedy_action:
                    break  # si π(a|s) = 0, terminamos
                
                # Política de comportamiento b: aleatoria uniforme
                prob_b = self.epsilon / self.nA + (1.0 - self.epsilon) 
                
                # Actualizar peso de importancia
                W *= 1.0 / prob_b  # π(a|s) = 1, así que solo dividimos por prob_b

                if W > 1e6:  # límite de estabilidad
                    break

            self.episode = []
            self.episode_return = 0.0
            self.episode_length = 0

    # disminuir epsilon para que al final vaya explorando menos y explotando más
    # porque ahí ya sabemos cuál es la mejor opción
    def decay_epsilon(self, episode_number):
        if self.decay:
            self.epsilon = min(1.0, 1000.0 / (episode_number + 1))
            # self.epsilon = max(0.01, 1.0 - episode_number * (0.99 / 8000))