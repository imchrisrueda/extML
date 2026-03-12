from .features import FourierFeatures, cartpole_features, tetris_features, N_TETRIS_FEATURES
from .sarsa_agent import SarsaLinearAgent

__all__ = [
    "FourierFeatures",
    "cartpole_features",
    "tetris_features",
    "N_TETRIS_FEATURES",
    "SarsaLinearAgent",
]

# El stack DQN/entrenamiento depende de torch, pero el flujo tabular no.
# Dejar estos imports como opcionales evita que `import entornos_complejos.src.tabular_taxi`
# falle en entornos donde solo se usan los experimentos tabulares.
try:
    from .dqn_agent import ReplayBuffer, QNetwork, DQNAgent
    from .training import (
        set_global_seed,
        cartpole_obs,
        tetris_obs_flat,
        train_sarsa_cartpole,
        train_sarsa_tetris,
        train_dqn_cartpole,
        train_dqn_tetris,
        run_multiseed,
        summary_stats,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "ReplayBuffer",
            "QNetwork",
            "DQNAgent",
            "set_global_seed",
            "cartpole_obs",
            "tetris_obs_flat",
            "train_sarsa_cartpole",
            "train_sarsa_tetris",
            "train_dqn_cartpole",
            "train_dqn_tetris",
            "run_multiseed",
            "summary_stats",
        ]
    )
