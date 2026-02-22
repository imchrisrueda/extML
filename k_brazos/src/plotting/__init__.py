"""
Module: plotting/__init__.py
Description: Contiene las importaciones y modulos/clases públicas del paquete plotting.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

# Importación de módulos o clases
from .plotting import plot_average_rewards, plot_optimal_selections, plot_arm_statistics, plot_regret

# Lista de módulos o clases públicas
__all__ = ['plot_average_rewards', 'plot_optimal_selections', 'plot_arm_statistics', 'plot_regret']
