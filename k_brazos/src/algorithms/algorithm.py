"""
Module: algorithms/algorithm.py
Description: Contiene la implementación abstracta para de diferentes algoritmos de selección de brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


from abc import ABC, abstractmethod
import numpy as np

class Algorithm(ABC):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo con k brazos.
        :param k: Número de brazos.
        """
        # Número de brazos
        self.k: int = k
        # Número de veces que se ha seleccionado cada brazo
        self.counts: np.ndarray = np.zeros(k, dtype=int)
        # Recompensa promedio estimada de cada brazo
        self.values: np.ndarray = np.zeros(k, dtype=float)

    @abstractmethod
    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política del algoritmo.
        :return: Índice del brazo seleccionado.
        """
        raise NotImplementedError("Este método debe ser implementado por la subclase.")

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza las recompensas promedio estimadas de cada brazo.
        :param chosen_arm: Índice del brazo que fue tirado.
        :param reward: Recompensa obtenida.
        """
        self.counts[chosen_arm] += 1  # Incrementa el conteo del brazo seleccionado

        n = self.counts[chosen_arm]  # Número de veces que el brazo seleccionado ha sido seleccionado
        value = self.values[chosen_arm]  # Valor actual del brazo seleccionado

        # Actualización incremental de la recompensa promedio
        # value = value + (reward - value) / n

        self.values[chosen_arm] = value + (reward - value) / n

    def reset(self):
        """
        Reinicia el estado del algoritmo (opcional).
        """
        self.counts = np.zeros(self.k, dtype=int)
        self.values = np.zeros(self.k, dtype=float)
