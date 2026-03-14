"""API pública de ``entornos_complejos.src``.

El paquete exporta una capa de compatibilidad para los módulos históricos y
la implementación consolidada usada por ``B_control_aprox.ipynb``.

Se usan imports perezosos para evitar cargar de forma innecesaria stacks que
no participan en el flujo actual del notebook.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FourierFeatures",
    "cartpole_features",
    "tetris_features",
    "N_TETRIS_FEATURES",
    "SarsaLinearAgent",
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

_EXPORT_MAP = {
    "FourierFeatures": (".features", "FourierFeatures"),
    "cartpole_features": (".features", "cartpole_features"),
    "tetris_features": (".features", "tetris_features"),
    "N_TETRIS_FEATURES": (".features", "N_TETRIS_FEATURES"),
    "SarsaLinearAgent": (".sarsa_agent", "SarsaLinearAgent"),
    "ReplayBuffer": (".dqn_agent", "ReplayBuffer"),
    "QNetwork": (".dqn_agent", "QNetwork"),
    "DQNAgent": (".dqn_agent", "DQNAgent"),
    "set_global_seed": (".training", "set_global_seed"),
    "cartpole_obs": (".training", "cartpole_obs"),
    "tetris_obs_flat": (".training", "tetris_obs_flat"),
    "train_sarsa_cartpole": (".training", "train_sarsa_cartpole"),
    "train_sarsa_tetris": (".training", "train_sarsa_tetris"),
    "train_dqn_cartpole": (".training", "train_dqn_cartpole"),
    "train_dqn_tetris": (".training", "train_dqn_tetris"),
    "run_multiseed": (".training", "run_multiseed"),
    "summary_stats": (".training", "summary_stats"),
}


def __getattr__(name: str):
    """Resuelve exports bajo demanda para reducir coste y acoplamiento."""
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
