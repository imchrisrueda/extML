"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo softmax (Boltzmann sobre Q) para el problema de los k-brazos.

Authors: García Echevarría, Aida; Rueda Ayala, Christian Andrés; Cuña Cabrera, Pablo Daniel
Date: 2026/02/15
"""

import numpy as np

from .algorithm import Algorithm


class Softmax(Algorithm):
    def __init__(self, k: int, temperature: float = 0.1):
        """
        Inicializa el algoritmo Softmax con exploración de Boltzmann sobre Q(a).

        :param k: Número de brazos.
        :param temperature: Temperatura usada para controlar la exploración.
                            Valores pequeños explotan más y valores grandes exploran más.
                            Se aplica sobre las estimaciones Q(a) almacenadas en self.values.
        """
        assert temperature > 0, "El parámetro temperature debe ser mayor que 0."

        super().__init__(k)
        self.temperature = temperature

    def _get_action_probabilities(self) -> np.ndarray:
        """
        Calcula la distribución de probabilidad Softmax sobre los valores Q estimados.

        :return: Vector de probabilidades de tamaño k.
        """
        scaled_values = self.values / self.temperature

        # Estabilización numérica para evitar overflows en exp.
        stabilized = scaled_values - np.max(scaled_values)
        exp_values = np.exp(stabilized)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def select_arm(self) -> int:
        """
        Selecciona un brazo muestreando desde la política Softmax de Boltzmann sobre Q(a).

        :return: Índice del brazo seleccionado.
        """
        probabilities = self._get_action_probabilities()
        chosen_arm = np.random.choice(self.k, p=probabilities)
        return int(chosen_arm)
