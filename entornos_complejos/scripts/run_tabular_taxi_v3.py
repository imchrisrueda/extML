from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entornos_complejos.src.tabular_taxi import (  # noqa: E402
    ExperimentConfig,
    METHOD_LABELS,
    METHOD_ORDER,
    run_full_experiment,
    save_results,
)


RESULTS_PATH = ROOT / "entornos_complejos" / "artifacts" / "tabular_taxi_v3_results.json"


def build_summary_frame(results: dict) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        metrics = results["summary"][method]
        rows.append(
            {
                "Metodo": METHOD_LABELS[method],
                "Train reward ultimos 1000": metrics["train_reward_last_window_mean"],
                "Train reward std semillas": metrics["train_reward_last_window_std"],
                "Train longitud ultimos 1000": metrics["train_length_last_window_mean"],
                "Train exito ultimos 1000": metrics["train_success_last_window_mean"],
                "Eval greedy final": metrics["final_eval_reward_mean"],
                "Eval greedy std semillas": metrics["final_eval_reward_std_across_seeds"],
                "Eval greedy longitud": metrics["final_eval_length_mean"],
                "Eval greedy exito": metrics["final_eval_success_rate_mean"],
            }
        )
    return pd.DataFrame(rows).set_index("Metodo")


def main() -> None:
    config = ExperimentConfig()
    print("Protocolo experimental")
    print(config)
    print()

    results = run_full_experiment(config, verbose=True)
    save_results(results, RESULTS_PATH)

    df = build_summary_frame(results)
    print("\nResumen agregado por metodo")
    print(df.to_string(float_format=lambda value: f"{value:0.3f}"))
    print()
    print("Ranking por evaluacion greedy final")
    for idx, method in enumerate(results["ranking"], start=1):
        metrics = results["summary"][method]
        print(
            f"{idx}. {METHOD_LABELS[method]}: "
            f"{metrics['final_eval_reward_mean']:.3f} +/- "
            f"{metrics['final_eval_reward_std_across_seeds']:.3f}"
        )
    print()
    print(f"Resultados guardados en: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
