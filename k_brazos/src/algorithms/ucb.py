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



class UCB2(Algorithm):
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2 (Auer et al., 2002) para un bandido de k brazos.

        UCB2 divide el proceso en épocas. En cada nueva época:
        1) Selecciona el brazo j que maximiza: x̄_j + a_{n, r_j}
        2) Juega el brazo j exactamente: τ(r_j+1) - τ(r_j) veces
        3) Incrementa r_j

        :param k: Número de brazos del bandido.
        :param alpha: Parámetro de UCB2 (0 < alpha < 1) que controla el crecimiento de τ(r).
        """
        assert 0.0 < alpha < 1.0, "El parámetro alpha debe estar en (0, 1)."
        super().__init__(k)
        self.alpha = float(alpha)

        # r_i: número de épocas completadas por cada brazo i (inicialmente 0)
        self.epochs = np.zeros(self.k, dtype=int)

        # Estado de la época actual
        self._current_arm = None
        self._plays_left_in_epoch = 0

    def _tau(self, r: int) -> int:
        """
        τ(r) = ceil((1 + alpha)^r)   (ver Auer et al., 2002).
        """
        return int(np.ceil((1.0 + self.alpha) ** r))

    def _a(self, n: int, r: int) -> float:
        """
        a_{n,r} = sqrt( ((1 + alpha) * ln(e*n / τ(r))) / (2 * τ(r)) )
        """
        tau_r = self._tau(r)
        # n >= 1 en el momento de uso (tras el warm-up). Aun así protegemos log.
        inside_log = (np.e * n) / tau_r
        return float(np.sqrt(((1.0 + self.alpha) * np.log(inside_log)) / (2.0 * tau_r)))

    def _get_ucb2_scores(self, n: int) -> np.ndarray:
        """
        Devuelve x̄_i + a_{n, r_i} para cada brazo i (asumiendo counts>0).
        """
        scores = np.empty(self.k, dtype=float)
        for i in range(self.k):
            scores[i] = self.values[i] + self._a(n, int(self.epochs[i]))
        return scores

    def select_arm(self) -> int:
        """
        Selecciona un brazo según UCB2:
        - Warm-up: cada brazo se selecciona al menos una vez.
        - Si estamos dentro de una época, repite el mismo brazo hasta completarla.
        - Si inicia una nueva época, elige el brazo con mayor índice UCB2 y fija su longitud.
        """
        # 1) Warm-up: probar cada brazo una vez (como en la inicialización del paper)
        for i in range(self.k):
            if self.counts[i] == 0:
                return int(i)

        # 2) Si aún quedan jugadas de la época actual, mantener el mismo brazo
        if self._plays_left_in_epoch > 0 and self._current_arm is not None:
            self._plays_left_in_epoch -= 1
            return int(self._current_arm)

        # 3) Empezar nueva época
        n = int(np.sum(self.counts))  # número total de jugadas realizadas hasta ahora (n >= k >= 1)

        scores = self._get_ucb2_scores(n)
        j = int(np.argmax(scores))

        # Longitud de la época del brazo j: τ(r_j+1) - τ(r_j)
        rj = int(self.epochs[j])
        epoch_len = self._tau(rj + 1) - self._tau(rj)

        # Fijar estado de época: vamos a jugar j 'epoch_len' veces (esta llamada cuenta como 1)
        self._current_arm = j
        self._plays_left_in_epoch = max(epoch_len - 1, 0)

        # Incrementar contador de épocas del brazo elegido
        self.epochs[j] += 1

        return int(j)