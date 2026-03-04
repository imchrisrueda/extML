"""
Module: algorithms/ucb.py
Description: Implementación del algoritmo ucb para el problema de los k-brazos.

Authors: García Echevarría, Aida; Rueda Ayala, Christian Andrés; Cuña Cabrera, Pablo Daniel
Date: 2026/02/28
"""

import numpy as np

from .algorithm import Algorithm


class UCB1(Algorithm):
    def __init__(self, k: int, c: float = np.sqrt(2.0)):
        """
        Inicializa el algoritmo UCB (Auer et al., 2002) para un bandido de k brazos.

        En la forma general:
            UCB(a) = Q(a) + c * sqrt( ln(t) / N(a) )

        Nota: c = sqrt(2) equivale a la forma clásica:
            Q(a) + sqrt( (2 ln t) / N(a) )

        :param k: Número de brazos del bandido.
        :param c: Constante de exploración (c > 0).
        """
        assert c > 0, "El parámetro c debe ser mayor que 0."
        super().__init__(k)
        self.c = float(c)

    def _get_ucb1_scores(self) -> np.ndarray:
        """
        Calcula UCB(a) = Q(a) + c * sqrt( ln(t) / N(a) ) para cada brazo.

        Importante: self.counts[i] > 0 para todo i para evitar divisiones entre 0.

        :return: Vector de scores UCB de tamaño k.
        """
        t = np.sum(self.counts)
        ucb_scores = self.values + self.c * np.sqrt(np.log(t) / self.counts)
        return ucb_scores

    def select_arm(self) -> int:
        """
        Selecciona un brazo según UCB:
        - Primero verifica que cada brazo haya sido seleccionado al menos una vez.
        - Luego elige el brazo que maximiza UCB(a) (Auer et al., 2002) sobre Q(a).
        """
        chosen_arm = None

        # Buscamos un brazo no probado
        for i in range(self.k):
            if self.counts[i] == 0:
                chosen_arm = i
                break

        # Si todos fueron probados, aplicamos UCB
        if chosen_arm is None:
            scores = self._get_ucb1_scores()
            chosen_arm = int(np.argmax(scores))

        return int(chosen_arm)



import numpy as np

class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2 (Auer et al., 2002) para un bandido de k brazos.

        UCB2 organiza la exploración en épocas. En cada nueva época:
        1) Selecciona el brazo j que maximiza: Q(j) + a(n, r_j)
        2) Acciona el brazo j una cantidad fija de veces asociada a esa época.
        3) Incrementa el contador de épocas r_j del brazo seleccionado.

        :param k: Número de brazos del bandido.
        :param alpha: Parámetro de UCB2 (0 < alpha < 1) que controla el crecimiento de la función tau(r).
        """
        assert 0.0 < alpha < 1.0, "El parámetro alpha debe estar en (0, 1)."
        super().__init__(k)

        self.alpha = float(alpha)

        # r_i: número de épocas completadas por cada brazo i (inicialmente 0)
        self.epocas = np.zeros(self.k, dtype=int)

        # Estado interno para mantener el brazo durante una época
        self._brazo_actual = None
        self._restantes_en_epoca = 0

    def _tau(self, r: int) -> int:
        """
        Función tau(r) que define el tamaño acumulado hasta la época r:
        tau(r) = ceil((1 + alpha)^r)

        :param r: Índice de época (entero no negativo).
        :return: Valor entero de tau(r).
        """
        return int(np.ceil((1.0 + self.alpha) ** r))

    def _a(self, n: int, r: int) -> float:
        """
        Término de confianza a(n, r) utilizado por UCB2 (Auer et al., 2002):

        a(n, r) = sqrt( ((1 + alpha) * ln(e*n / tau(r))) / (2 * tau(r+1)) )

        Nota: en el denominador aparece tau(r+1), no tau(r).

        :param n: Número total de acciones realizadas hasta el momento.
        :param r: Número de épocas completadas por el brazo.
        :return: Valor del término de confianza.
        """
        tau_r = self._tau(r)
        tau_r1 = self._tau(r + 1)

        # En la práctica, tau_r <= n siempre debería cumplirse si el algoritmo está funcionando bien,
        # pero por robustez evitamos log() de valores <= 0.
        ratio = (np.e * n) / tau_r
        ratio = max(ratio, 1.0 + 1e-12)

        return float(np.sqrt(((1.0 + self.alpha) * np.log(ratio)) / (2.0 * tau_r1)))

    def _get_ucb2_scores(self, n: int) -> np.ndarray:
        """
        Calcula el índice de UCB2 para cada brazo:
        UCB2(i) = Q(i) + a(n, r_i)

        :param n: Número total de acciones realizadas hasta el momento.
        :return: Vector de índices UCB2 de tamaño k.
        """
        ucb2_scores = np.empty(self.k, dtype=float)
        for i in range(self.k):
            ucb2_scores[i] = self.values[i] + self._a(n, int(self.epocas[i]))
        return ucb2_scores

    def select_arm(self) -> int:
        """
        Selecciona un brazo según UCB2:
        - Primero verifica que cada brazo haya sido seleccionado al menos una vez.
        - Si estamos dentro de una época, mantiene el mismo brazo hasta completarla.
        - Si se inicia una nueva época, elige el brazo que maximiza UCB2(i) y fija su duración.
        """
        # Verificación inicial: probar cada brazo al menos una vez
        for i in range(self.k):
            if self.counts[i] == 0:
                return int(i)

        # Si aún quedan acciones dentro de la época actual, repetir el mismo brazo
        if self._restantes_en_epoca > 0 and self._brazo_actual is not None:
            self._restantes_en_epoca -= 1
            return int(self._brazo_actual)

        # Iniciar una nueva época
        n = int(np.sum(self.counts))  # total de acciones realizadas hasta ahora (n >= k)

        scores = self._get_ucb2_scores(n)
        brazo_elegido = int(np.argmax(scores))

        # Duración de la época para el brazo elegido: tau(r+1) - tau(r)
        r_brazo = int(self.epocas[brazo_elegido])
        duracion_epoca = self._tau(r_brazo + 1) - self._tau(r_brazo)

        # Configurar el estado de época (esta llamada cuenta como la primera acción)
        self._brazo_actual = brazo_elegido
        self._restantes_en_epoca = max(duracion_epoca - 1, 0)

        # Incrementar el contador de épocas del brazo elegido
        self.epocas[brazo_elegido] += 1

        return int(brazo_elegido)