"""
Module: arms/bandit.py
Description: Contains the implementation of the Bandit class for the k-armed bandit problem.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""


# bandit.py
from typing import List

import numpy as np

from arms import Arm


class Bandit:
    def __init__(self, arms: List[Arm]):
        """
        Initializes the bandit with a list of arms.

        :param arms: List of instances of classes derived from Arm.
        :type arms: list of Arm
        """
        self.arms = arms
        self.k = len(arms)
        self.expected_rewards = self.get_expected_rewards()
        self.optimal_arm = self.get_optimal_arm()

    def pull_arm(self, index: int) -> float:
        """
        Pulls a specific arm and returns the reward.

        :param index: Index of the arm to pull (0 to k-1).
        :return: Reward obtained from the arm.
        :raises IndexError: If the index is out of the valid range.
        """
        if index < 0 or index >= self.k:
            raise IndexError("Arm index out of range.")

        reward = self.arms[index].pull()
        return reward

    def get_optimal_arm(self) -> int:
        """
        Identifies the arm with the highest expected reward.

        :return: Index of the optimal arm.
        """

        optimal_arm = np.argmax(self.expected_rewards)
        return optimal_arm

    def get_expected_rewards(self) -> List[float]:
        """
        Returns the reward of each arm in the bandit.

        :return: List of rewards for each arm.
        :rtype: list of float or int
        """
        rewards = [arm.get_expected_value() for arm in self.arms]
        return rewards

    def get_expected_value(self, numer_arm):
        return self.arms[numer_arm].get_expected_value()

    def __len__(self):
        """
        Returns the number of arms in the bandit.
        :return:
        """
        return self.k

    def __str__(self):
        """
        String representation of the bandit showing the types of arms.

        :return: Detailed description of the bandit and its arms.
        :rtype: str
        """
        arms_description = ", ".join([str(arm) for arm in self.arms])
        return f"Bandit with {self.k} arms: {arms_description}"
