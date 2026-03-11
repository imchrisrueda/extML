"""
Experimentos tabulares comparables para Taxi-v3.

Incluye:
- Monte Carlo on-policy every-visit
- Monte Carlo off-policy con Weighted Importance Sampling
- SARSA tabular
- Q-Learning tabular

El objetivo es ejecutar los cuatro métodos bajo exactamente el mismo
protocolo experimental y separar claramente:
- métricas de entrenamiento de la política que interactúa con el entorno
- métricas de evaluación greedy de la política aprendida
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Callable

import gymnasium as gym
import numpy as np


METHOD_ORDER = (
    "mc_on_policy",
    "mc_off_policy_weighted",
    "sarsa",
    "q_learning",
)

METHOD_LABELS = {
    "mc_on_policy": "MC On-Policy (Every-Visit)",
    "mc_off_policy_weighted": "MC Off-Policy (Weighted IS)",
    "sarsa": "SARSA",
    "q_learning": "Q-Learning",
}

METHOD_COLORS = {
    "mc_on_policy": "#1565C0",
    "mc_off_policy_weighted": "#B71C1C",
    "sarsa": "#F57C00",
    "q_learning": "#00695C",
}


@dataclass(frozen=True)
class ExperimentConfig:
    env_id: str = "Taxi-v3"
    n_episodes: int = 10_000
    gamma: float = 0.99
    alpha_td: float = 0.10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9995
    eval_every: int = 250
    eval_episodes: int = 50
    final_eval_episodes: int = 200
    rolling_window: int = 200
    summary_window: int = 1_000
    seeds: tuple[int, ...] = (123, 231, 777, 2024, 31415)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["seeds"] = list(self.seeds)
        return data


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def epsilon_for_episode(config: ExperimentConfig, episode_idx: int) -> float:
    epsilon = config.epsilon_start * (config.epsilon_decay ** episode_idx)
    return float(max(config.epsilon_end, epsilon))


def training_reset_seed(run_seed: int, episode_idx: int) -> int:
    return int(run_seed * 100_000 + episode_idx)


def evaluation_reset_seeds(run_seed: int, n_eval: int, offset: int) -> list[int]:
    base = run_seed * 1_000_000 + offset
    return [base + idx for idx in range(n_eval)]


def greedy_action(Q: np.ndarray, state: int) -> int:
    return int(np.argmax(Q[state]))


def epsilon_greedy_probs(
    Q: np.ndarray,
    state: int,
    epsilon: float,
) -> np.ndarray:
    n_actions = Q.shape[1]
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    probs[greedy_action(Q, state)] += 1.0 - epsilon
    return probs


def sample_epsilon_greedy(
    Q: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[int, float]:
    probs = epsilon_greedy_probs(Q, state, epsilon)
    action = int(rng.choice(Q.shape[1], p=probs))
    return action, float(probs[action])


def evaluate_greedy_policy(
    config: ExperimentConfig,
    Q: np.ndarray,
    reset_seeds: list[int],
) -> dict:
    env = gym.make(config.env_id)
    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []

    for seed in reset_seeds:
        state, _ = env.reset(seed=seed)
        done = False
        total_reward = 0.0
        steps = 0
        episode_success = 0

        while not done:
            action = greedy_action(Q, state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if done:
                episode_success = int(terminated)

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(episode_success)

    env.close()

    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }


def _finalize_run(
    config: ExperimentConfig,
    seed: int,
    rewards: list[float],
    lengths: list[int],
    successes: list[int],
    eval_points: list[int],
    eval_reward_mean: list[float],
    eval_reward_std: list[float],
    eval_length_mean: list[float],
    eval_success_rate: list[float],
    Q: np.ndarray,
) -> dict:
    final_eval = evaluate_greedy_policy(
        config,
        Q,
        evaluation_reset_seeds(seed, config.final_eval_episodes, offset=900_000),
    )
    tail = slice(-config.summary_window, None)
    return {
        "seed": seed,
        "rewards": rewards,
        "lengths": lengths,
        "successes": successes,
        "eval_episodes": eval_points,
        "eval_reward_mean": eval_reward_mean,
        "eval_reward_std": eval_reward_std,
        "eval_length_mean": eval_length_mean,
        "eval_success_rate": eval_success_rate,
        "train_reward_last_window": float(np.mean(rewards[tail])),
        "train_length_last_window": float(np.mean(lengths[tail])),
        "train_success_last_window": float(np.mean(successes[tail])),
        "final_eval_reward_mean": final_eval["reward_mean"],
        "final_eval_reward_std": final_eval["reward_std"],
        "final_eval_length_mean": final_eval["length_mean"],
        "final_eval_success_rate": final_eval["success_rate"],
    }


def run_mc_on_policy(config: ExperimentConfig, seed: int) -> dict:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(config.env_id)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_sum = np.zeros((n_states, n_actions), dtype=np.float64)
    returns_count = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    eval_points: list[int] = []
    eval_reward_mean: list[float] = []
    eval_reward_std: list[float] = []
    eval_length_mean: list[float] = []
    eval_success_rate: list[float] = []
    periodic_eval_seeds = evaluation_reset_seeds(seed, config.eval_episodes, offset=100_000)

    for episode_idx in range(config.n_episodes):
        epsilon = epsilon_for_episode(config, episode_idx)
        state, _ = env.reset(seed=training_reset_seed(seed, episode_idx))
        done = False
        total_reward = 0.0
        steps = 0
        episode_success = 0
        trajectory: list[tuple[int, int, float]] = []

        while not done:
            action, _ = sample_epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, reward))
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if done:
                episode_success = int(terminated)

        G = 0.0
        for state, action, reward in reversed(trajectory):
            G = config.gamma * G + reward
            returns_sum[state, action] += G
            returns_count[state, action] += 1.0
            Q[state, action] = returns_sum[state, action] / returns_count[state, action]

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(episode_success)

        if (episode_idx + 1) % config.eval_every == 0:
            metrics = evaluate_greedy_policy(config, Q, periodic_eval_seeds)
            eval_points.append(episode_idx + 1)
            eval_reward_mean.append(metrics["reward_mean"])
            eval_reward_std.append(metrics["reward_std"])
            eval_length_mean.append(metrics["length_mean"])
            eval_success_rate.append(metrics["success_rate"])

    env.close()
    return _finalize_run(
        config,
        seed,
        rewards,
        lengths,
        successes,
        eval_points,
        eval_reward_mean,
        eval_reward_std,
        eval_length_mean,
        eval_success_rate,
        Q,
    )


def run_mc_off_policy_weighted(config: ExperimentConfig, seed: int) -> dict:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(config.env_id)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    C = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    eval_points: list[int] = []
    eval_reward_mean: list[float] = []
    eval_reward_std: list[float] = []
    eval_length_mean: list[float] = []
    eval_success_rate: list[float] = []
    periodic_eval_seeds = evaluation_reset_seeds(seed, config.eval_episodes, offset=100_000)

    for episode_idx in range(config.n_episodes):
        epsilon = epsilon_for_episode(config, episode_idx)
        state, _ = env.reset(seed=training_reset_seed(seed, episode_idx))
        done = False
        total_reward = 0.0
        steps = 0
        episode_success = 0
        trajectory: list[tuple[int, int, float, float]] = []

        while not done:
            action, prob_b = sample_epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, reward, prob_b))
            state = next_state
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if done:
                episode_success = int(terminated)

        G = 0.0
        W = 1.0
        for state, action, reward, prob_b in reversed(trajectory):
            G = config.gamma * G + reward
            C[state, action] += W
            Q[state, action] += (W / C[state, action]) * (G - Q[state, action])

            if action != greedy_action(Q, state):
                break

            W *= 1.0 / prob_b

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(episode_success)

        if (episode_idx + 1) % config.eval_every == 0:
            metrics = evaluate_greedy_policy(config, Q, periodic_eval_seeds)
            eval_points.append(episode_idx + 1)
            eval_reward_mean.append(metrics["reward_mean"])
            eval_reward_std.append(metrics["reward_std"])
            eval_length_mean.append(metrics["length_mean"])
            eval_success_rate.append(metrics["success_rate"])

    env.close()
    return _finalize_run(
        config,
        seed,
        rewards,
        lengths,
        successes,
        eval_points,
        eval_reward_mean,
        eval_reward_std,
        eval_length_mean,
        eval_success_rate,
        Q,
    )


def run_sarsa(config: ExperimentConfig, seed: int) -> dict:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(config.env_id)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    eval_points: list[int] = []
    eval_reward_mean: list[float] = []
    eval_reward_std: list[float] = []
    eval_length_mean: list[float] = []
    eval_success_rate: list[float] = []
    periodic_eval_seeds = evaluation_reset_seeds(seed, config.eval_episodes, offset=100_000)

    for episode_idx in range(config.n_episodes):
        epsilon = epsilon_for_episode(config, episode_idx)
        state, _ = env.reset(seed=training_reset_seed(seed, episode_idx))
        action, _ = sample_epsilon_greedy(Q, state, epsilon, rng)
        done = False
        total_reward = 0.0
        steps = 0
        episode_success = 0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                td_target = reward
                episode_success = int(terminated)
            else:
                next_action, _ = sample_epsilon_greedy(Q, next_state, epsilon, rng)
                td_target = reward + config.gamma * Q[next_state, next_action]

            Q[state, action] += config.alpha_td * (td_target - Q[state, action])
            total_reward += reward
            steps += 1

            if not done:
                state, action = next_state, next_action

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(episode_success)

        if (episode_idx + 1) % config.eval_every == 0:
            metrics = evaluate_greedy_policy(config, Q, periodic_eval_seeds)
            eval_points.append(episode_idx + 1)
            eval_reward_mean.append(metrics["reward_mean"])
            eval_reward_std.append(metrics["reward_std"])
            eval_length_mean.append(metrics["length_mean"])
            eval_success_rate.append(metrics["success_rate"])

    env.close()
    return _finalize_run(
        config,
        seed,
        rewards,
        lengths,
        successes,
        eval_points,
        eval_reward_mean,
        eval_reward_std,
        eval_length_mean,
        eval_success_rate,
        Q,
    )


def run_q_learning(config: ExperimentConfig, seed: int) -> dict:
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(config.env_id)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    rewards: list[float] = []
    lengths: list[int] = []
    successes: list[int] = []
    eval_points: list[int] = []
    eval_reward_mean: list[float] = []
    eval_reward_std: list[float] = []
    eval_length_mean: list[float] = []
    eval_success_rate: list[float] = []
    periodic_eval_seeds = evaluation_reset_seeds(seed, config.eval_episodes, offset=100_000)

    for episode_idx in range(config.n_episodes):
        epsilon = epsilon_for_episode(config, episode_idx)
        state, _ = env.reset(seed=training_reset_seed(seed, episode_idx))
        done = False
        total_reward = 0.0
        steps = 0
        episode_success = 0

        while not done:
            action, _ = sample_epsilon_greedy(Q, state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            td_target = reward if done else reward + config.gamma * np.max(Q[next_state])
            Q[state, action] += config.alpha_td * (td_target - Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1
            if done:
                episode_success = int(terminated)

        rewards.append(total_reward)
        lengths.append(steps)
        successes.append(episode_success)

        if (episode_idx + 1) % config.eval_every == 0:
            metrics = evaluate_greedy_policy(config, Q, periodic_eval_seeds)
            eval_points.append(episode_idx + 1)
            eval_reward_mean.append(metrics["reward_mean"])
            eval_reward_std.append(metrics["reward_std"])
            eval_length_mean.append(metrics["length_mean"])
            eval_success_rate.append(metrics["success_rate"])

    env.close()
    return _finalize_run(
        config,
        seed,
        rewards,
        lengths,
        successes,
        eval_points,
        eval_reward_mean,
        eval_reward_std,
        eval_length_mean,
        eval_success_rate,
        Q,
    )


RUNNERS: dict[str, Callable[[ExperimentConfig, int], dict]] = {
    "mc_on_policy": run_mc_on_policy,
    "mc_off_policy_weighted": run_mc_off_policy_weighted,
    "sarsa": run_sarsa,
    "q_learning": run_q_learning,
}


def create_results_skeleton(config: ExperimentConfig) -> dict:
    return {
        "config": config.to_dict(),
        "method_order": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "method_colors": METHOD_COLORS,
        "runs": {method: [] for method in METHOD_ORDER},
    }


def aggregate_results(results: dict) -> dict:
    summary: dict[str, dict] = {}
    curves: dict[str, dict] = {}
    available_methods: list[str] = []

    for method in METHOD_ORDER:
        runs = results["runs"][method]
        if not runs:
            continue
        available_methods.append(method)
        reward_last = np.array([run["train_reward_last_window"] for run in runs], dtype=np.float64)
        length_last = np.array([run["train_length_last_window"] for run in runs], dtype=np.float64)
        success_last = np.array([run["train_success_last_window"] for run in runs], dtype=np.float64)
        final_reward = np.array([run["final_eval_reward_mean"] for run in runs], dtype=np.float64)
        final_length = np.array([run["final_eval_length_mean"] for run in runs], dtype=np.float64)
        final_success = np.array([run["final_eval_success_rate"] for run in runs], dtype=np.float64)

        eval_reward_stack = np.array([run["eval_reward_mean"] for run in runs], dtype=np.float64)
        eval_success_stack = np.array([run["eval_success_rate"] for run in runs], dtype=np.float64)
        eval_length_stack = np.array([run["eval_length_mean"] for run in runs], dtype=np.float64)

        summary[method] = {
            "train_reward_last_window_mean": float(np.mean(reward_last)),
            "train_reward_last_window_std": float(np.std(reward_last)),
            "train_length_last_window_mean": float(np.mean(length_last)),
            "train_length_last_window_std": float(np.std(length_last)),
            "train_success_last_window_mean": float(np.mean(success_last)),
            "train_success_last_window_std": float(np.std(success_last)),
            "final_eval_reward_mean": float(np.mean(final_reward)),
            "final_eval_reward_std_across_seeds": float(np.std(final_reward)),
            "final_eval_length_mean": float(np.mean(final_length)),
            "final_eval_length_std_across_seeds": float(np.std(final_length)),
            "final_eval_success_rate_mean": float(np.mean(final_success)),
            "final_eval_success_rate_std_across_seeds": float(np.std(final_success)),
        }

        curves[method] = {
            "eval_episodes": runs[0]["eval_episodes"],
            "eval_reward_mean": np.mean(eval_reward_stack, axis=0).tolist(),
            "eval_reward_std_across_seeds": np.std(eval_reward_stack, axis=0).tolist(),
            "eval_success_rate_mean": np.mean(eval_success_stack, axis=0).tolist(),
            "eval_success_rate_std_across_seeds": np.std(eval_success_stack, axis=0).tolist(),
            "eval_length_mean": np.mean(eval_length_stack, axis=0).tolist(),
            "eval_length_std_across_seeds": np.std(eval_length_stack, axis=0).tolist(),
        }

    ranking = sorted(
        available_methods,
        key=lambda method: summary[method]["final_eval_reward_mean"],
        reverse=True,
    )
    return {"summary": summary, "curves": curves, "ranking": ranking}


def finalize_results(results: dict) -> dict:
    aggregated = aggregate_results(results)
    results["summary"] = aggregated["summary"]
    results["curves"] = aggregated["curves"]
    results["ranking"] = aggregated["ranking"]
    return results


def run_method_experiment(
    config: ExperimentConfig,
    method: str,
    verbose: bool = False,
) -> dict:
    if method not in RUNNERS:
        raise ValueError(f"Metodo desconocido: {method}")

    results = create_results_skeleton(config)
    runner = RUNNERS[method]
    for seed in config.seeds:
        if verbose:
            print(f"[run] {METHOD_LABELS[method]} | seed={seed}")
        results["runs"][method].append(runner(config, seed))
    return finalize_results(results)


def combine_method_results(config: ExperimentConfig, method_results: dict[str, dict]) -> dict:
    results = create_results_skeleton(config)
    for method in METHOD_ORDER:
        method_result = method_results.get(method)
        if method_result is not None:
            results["runs"][method] = method_result["runs"][method]
    return finalize_results(results)


def run_full_experiment(config: ExperimentConfig, verbose: bool = False) -> dict:
    results = create_results_skeleton(config)

    for method in METHOD_ORDER:
        runner = RUNNERS[method]
        for seed in config.seeds:
            if verbose:
                print(f"[run] {METHOD_LABELS[method]} | seed={seed}")
            results["runs"][method].append(runner(config, seed))

    return finalize_results(results)


def save_results(results: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


def load_results(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)
