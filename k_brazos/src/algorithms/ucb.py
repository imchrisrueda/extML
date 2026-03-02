"""
Module: algorithms/ucb.py
Description: Implementación del algoritmo ucb para el problema de los k-brazos.

Authors: García Echevarría, Aida; Rueda Ayala, Christian Andrés; Cuña Cabrera, Pablo Daniel
Date: 2026/02/28
"""

import numpy as np

from .algorithm import Algorithm


class UCB1(Algorithm):
    def __init__(self, k: int):
        """
        Inicializa el algoritmo UCB1 (Auer et al., 2002) para un bandido de k brazos..

        :param k: Número de brazos del bandido.
        
        """
        
        super().__init__(k)

    def _get_ucb1_scores(self) -> np.ndarray:
        """
        Calcula UCB1(a) = Q(a) + sqrt( (2 ln t) / N(a) ) para cada brazo.

        Importante: self.counts[i] > 0 para todo i para evitar divisiones entre 0.

        :return: Vector de scores UCB1 de tamaño k.
        """
        t = np.sum(self.counts)
        ucb1_scores = self.values + np.sqrt((2.0 * np.log(t)) / self.counts)
        return ucb1_scores        


    def select_arm(self) -> int:
        """
        Selecciona un brazo según UCB1:
        - Primero verifica que cada brazo haya sido seleccionado al menos una vez.
        - Luego elige el brazo que maximiza UCB1(a) de Auer et al sobre Q(a)..
        """
        chosen_arm = None

        # Buscamos un brazo no probado
        for i in range(self.k):
            if self.counts[i] == 0:
                chosen_arm = i
                break

        # Si todos fueron probados, aplicamos UCB1
        if chosen_arm is None:
            scores = self._get_ucb1_scores()
            chosen_arm = int(np.argmax(scores))

        return int(chosen_arm)
