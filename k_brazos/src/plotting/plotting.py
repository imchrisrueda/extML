"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.

Author: Luis Daniel Hernández Molinero y modificado por los alumnos Aida García Echevarría; Christian Andrés Rueda Ayala; Pablo Daniel Cuña Cabrera
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ..algorithms import Algorithm, EpsilonGreedy


def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    # elif isinstance(algo, OtroAlgoritmo):
    #     label += f" (parametro={algo.parametro})"
    # Añadir más condiciones para otros algoritmos aquí
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label

#se ha añadido el parámetro distribution_name para poder indicar en el título del gráfico a qué distribución pertenece
def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm], distribution_name: str = ""):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param distribution_name: Tipo de distribución que se está ejecutando
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)

    title = "Recompensa Promedio vs Pasos de Tiempo"
    if distribution_name:
        title += f" ({distribution_name})"
    plt.title(title, fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm], distribution_name: str = ""):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param distribution_name: Tipo de distribución que se está ejecutando
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), optimal_selections[idx] * 100, label=label, linewidth=2) #x100 para obtener el porcentaje

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Selección del Brazo Óptimo', fontsize=14)
    
    title = "Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo"
    if distribution_name:
        title += f" ({distribution_name})"

    plt.title(title, fontsize=16)

    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], distribution_name: str = "", *args): 
    """ Genera la gráfica de Regret Acumulado vs Pasos de Tiempo
    
    :param steps: Número de pasos de tiempo. 
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos). 
    :param algorithms: Lista de instancias de algoritmos comparados. 
    :param distribution_name: Tipo de distribución que se está ejecutando
    :param args: Opcional. Parámetros que consideres. P.e. la cota teórica Cte * ln(T). 
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Regret Acumulado ', fontsize=14)
   
    title = "Regret Acumulado vs Pasos de Tiempo"
    if distribution_name:
        title += f" ({distribution_name})"

    plt.title(title, fontsize=16)

    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()


def plot_arm_statistics(arm_stats, algorithms: List[Algorithm], distribution_name: str = "",*args): 
    """ Genera gráficas separadas de Selección de Arms: Ganancias vs Pérdidas para cada algoritmo. 
    
    :param arm_stats: Lista (de diccionarios) con estadísticas de cada brazo por algoritmo. 
    :param algorithms: Lista de instancias de algoritmos comparados. 
    :param distribution_name: Tipo de distribución que se está ejecutando
    :param args: Opcional. Parámetros que consideres 
    """

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms): # Para cada algoritmo ejecutado
        label = get_algorithm_label(algo)
        stats = arm_stats[idx] # Cojo las estadísticas de ese algoritmo concreto

        arms = list(stats.keys())  # Las estadísticas van a ser una lista con la recompensa, los brazos seleccionados y si era óptimo o no
        # Me quedo con cada elemento dentro de las estadísticas
        avg_rewards = [stats[a]["avg_reward"] for a in arms] 
        selections = [stats[a]["selected"] for a in arms]
        is_optimal = [stats[a]["is_optimal"] for a in arms]

        colors = ["tab:green" if opt else "tab:red" for opt in is_optimal] # En verde el óptimo y si no en rojo

        plt.figure(figsize=(12, 6))
        bars = plt.bar(arms, avg_rewards, color=colors)

        # Etiquetas
        for bar, sel in zip(bars, selections):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"n={sel}",
                ha="center",
                va="bottom",
                fontsize=10
            )

        plt.xlabel("Brazo", fontsize=14)
        plt.ylabel("Recompensa promedio estimada", fontsize=14)

        title = f"Estadísticas por brazo – {label}"
        if distribution_name:
            title += f" ({distribution_name})"

        plt.title(title, fontsize=16)

        # Leyenda para óptimo/no óptimo
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="tab:green", label="Brazo óptimo"),
            Patch(facecolor="tab:red", label="Brazo no óptimo")
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()