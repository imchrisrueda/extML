"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from .algorithm import Algorithm

class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon
        self.selected_arms = set()  # Conjunto para llevar el registro de brazos seleccionados

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.

        :return: índice del brazo seleccionado.
        """

        # Observa que para para epsilon=0 solo selecciona un brazo y no hace un primer recorrido por todos ellos.
        # ¿Podrías modificar el código para que funcione correctamente para epsilon=0?
        # modificar el código proque para epsilon = 0 devuelve siempre el brazo 0, solo explora ese brazo

        #para cuando epsilon = 0 se va a hacer : probar al menos una vez todos los brazo para después empezar a explotar el mejor.

        # Asegurar que cada brazo se pruebe al menos una vez
        # self.k es el número total de brazos disponibles
        # y self.selected_arms almacena los índices de los brazos que ya han sido seleccionados
        # por lo que si el conjunto de índices seleccionas es menor que el número total de brazos disponibles, significa que no se han seleccionado todos los brazos
        if len(self.selected_arms) < self.k:
            # Selecciona un brazo no probado
            chosen_arm = len(self.selected_arms)  # Escoge el próximo brazo no probado
            self.selected_arms.add(chosen_arm)  # Agrega el brazo al conjunto de seleccionados

        # si todos los brazos ya han sido seleccionados al menos una vez:
        # igual que antes
        else:
            if np.random.random() < self.epsilon:
                # Selecciona un brazo al azar
                chosen_arm = np.random.choice(self.k)
            else:
                # Selecciona el brazo con la recompensa promedio estimada más alta
                chosen_arm = np.argmax(self.values)

        return chosen_arm



