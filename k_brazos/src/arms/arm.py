"""
Module: arms/arm.py
Description: Contains the abstract class Arm, which defines the interface for the arms used in the bandit problem.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

from abc import ABC, abstractmethod


class Arm(ABC):

    @classmethod
    def generate_arms(cls, k: int):
        """
        Generates a list of arms with random parameters.

        :param k: Number of arms to generate.
        :return: List of arms.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def pull(self):
        """
        Generates a reward based on the arm's distribution.

        This method must be implemented by derived classes.

        :raises NotImplementedError: If not implemented in the subclass.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    @abstractmethod
    def get_expected_value(self) -> float:
        """
        Calculates and returns the expected value of the arm's reward.
        """
        raise NotImplementedError("This method must be implemented by the subclass.")
