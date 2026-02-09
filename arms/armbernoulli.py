"""
Module: arms/bernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the bernoulli distribution arm.

Date: 2025/02/07

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""
import numpy as np

from arms import Arm


class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución bernoulli.

        :param p: Probabilidad de éxito.
        """

        assert 0.0 <= p <= 1.0, "p debe estar en [0,1]"

        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución bernoulli.

        :return: Recompensa obtenida del brazo.
        """
        reward = np.random.binomial(1, self.p) #Es un caso particular de la binomial con n=1
        return reward

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución bernoulli.

        :return: Valor esperado de la distribución.
        """

        return self.p

    def __str__(self):
        """
        Representación en cadena del brazo bernoulli.

        :return: Descripción detallada del brazo bernoulli.
        """
        return f"ArmBernoulli(p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, p_min: float = 0.01, p_max: float = 0.99):
        """
        Genera k brazos bernoulli con probabilidades únicas en el rango [p_min, p_max].

        :param k: Número de brazos a generar.
        :param n: Número de ensayos.
        :param mu_min: Valor mínimo de la probabilidad de éxito.
        :param mu_max: Valor máximo de la probabilidad de éxito.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert p_min < p_max, "El valor de p_min debe ser menor que p_max."

        # Generar k- valores únicos de p con decimales
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)

        arms = [ArmBernoulli(p) for p in p_values]

        return arms
