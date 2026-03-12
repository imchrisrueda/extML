from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entornos_complejos.src.tabular_taxi import METHOD_LABELS  # noqa: E402


RESULTS_PATH = ROOT / "entornos_complejos" / "artifacts" / "tabular_taxi_v3_results.json"
NOTEBOOK_PATH = ROOT / "entornos_complejos" / "B-MC_v3.ipynb"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in text.splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in source.splitlines()],
    }


def method_commentary(method_label: str, metrics: dict, emphasis: str) -> str:
    return f"""**Resultados agregados validados**

- Reward medio de entrenamiento en los últimos 1000 episodios: `{fmt(metrics['train_reward_last_window_mean'])} +/- {fmt(metrics['train_reward_last_window_std'])}`
- Longitud media de entrenamiento en los últimos 1000 episodios: `{fmt(metrics['train_length_last_window_mean'])}`
- Tasa de éxito real en los últimos 1000 episodios: `{pct(metrics['train_success_last_window_mean'])}`
- Reward medio en evaluación greedy final: `{fmt(metrics['final_eval_reward_mean'])} +/- {fmt(metrics['final_eval_reward_std_across_seeds'])}`
- Longitud media en evaluación greedy final: `{fmt(metrics['final_eval_length_mean'])}`
- Tasa de éxito real en evaluación greedy final: `{pct(metrics['final_eval_success_rate_mean'])}`

{emphasis}
"""


def build_notebook(results: dict) -> dict:
    config = results["config"]
    summary = results["summary"]
    ranking = results["ranking"]

    mc_on = summary["mc_on_policy"]
    mc_off = summary["mc_off_policy_weighted"]
    sarsa = summary["sarsa"]
    ql = summary["q_learning"]

    ranking_lines = []
    for idx, method in enumerate(ranking, start=1):
        metrics = summary[method]
        ranking_lines.append(
            f"{idx}. {METHOD_LABELS[method]}: "
            f"{fmt(metrics['final_eval_reward_mean'])} +/- "
            f"{fmt(metrics['final_eval_reward_std_across_seeds'])}"
        )

    intro = """# Parte B – Métodos Tabulares (Versión 3)

## Objetivo del notebook

En esta versión 3 se reconstruye todo el experimento con un formato más explicativo y más cercano al estilo narrativo del notebook `v2`, pero manteniendo la disciplina experimental que faltaba allí.

La idea es ir método por método, explicar:

- qué intenta aprender cada algoritmo
- cómo se implementa en este trabajo
- qué métricas usamos para validarlo de verdad
- y qué dicen los resultados obtenidos al ejecutar el código

El entorno sigue siendo `Taxi-v3`, pero ahora la comparación es completamente justa: mismo protocolo, mismas semillas, misma planificación de `epsilon`, misma evaluación greedy y la misma definición de éxito para todos los métodos.
"""

    environment = """## Recordatorio del entorno Taxi-v3

`Taxi-v3` es un entorno discreto con:

- `500` estados posibles
- `6` acciones discretas
- recompensa `-1` por paso
- recompensa `+20` cuando el taxi deja correctamente al pasajero
- penalización `-10` en acciones ilegales

La tarea real del agente no es solo llegar al objetivo alguna vez, sino aprender una política eficiente. Por eso en este notebook se van a separar dos cosas:

- el comportamiento durante el entrenamiento
- la calidad de la política greedy aprendida

Esa separación es especialmente importante en Monte Carlo off-policy, porque la política de comportamiento y la política objetivo no son la misma.
"""

    protocol = f"""## Protocolo experimental común

Estas son las condiciones exactas usadas en la validación:

- Entorno: `Taxi-v3`
- Semillas: {config['seeds']}
- Episodios de entrenamiento por semilla: `{config['n_episodes']}`
- Factor de descuento: `gamma = {config['gamma']}`
- Exploración: `epsilon` desde `{config['epsilon_start']}` hasta `{config['epsilon_end']}` con decaimiento multiplicativo `{config['epsilon_decay']}`
- Evaluación greedy periódica: cada `{config['eval_every']}` episodios, usando `{config['eval_episodes']}` episodios de evaluación por punto
- Evaluación greedy final: `{config['final_eval_episodes']}` episodios por semilla
- Resumen de entrenamiento: últimos `{config['summary_window']}` episodios
- Ventana de suavizado para gráficas: `{config['rolling_window']}`
- SARSA y Q-Learning usan `alpha = {config['alpha_td']}`

Además, en esta implementación el éxito se mide con la señal real del entorno:

- `success = 1` cuando el episodio termina por `terminated`
- `success = 0` cuando el episodio termina por truncado

Esto evita el atajo de usar `reward > 0` como proxy, que en Taxi puede ser engañoso.
"""

    module_note = """## Cómo está organizado el código

La implementación de los métodos está en `entornos_complejos/src/tabular_taxi.py`. El notebook no duplica esa lógica: la importa y la ejecuta por secciones, una por algoritmo.

Esa decisión permite dos cosas:

- mantener el notebook legible y centrado en la explicación
- dejar una implementación reutilizable, comprobable y fácil de volver a ejecutar

En las celdas siguientes primero se fija la configuración, después se definen funciones auxiliares para resumir y graficar resultados, y luego se entrena cada método por separado.
"""

    imports = """from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "entornos_complejos").exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from entornos_complejos.src.tabular_taxi import (
    ExperimentConfig,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    combine_method_results,
    run_method_experiment,
    save_results,
)

RESULTS_PATH = ROOT / "entornos_complejos" / "artifacts" / "tabular_taxi_v3_results.json"

config_obj = ExperimentConfig(
    env_id="Taxi-v3",
    n_episodes=10000,
    gamma=0.99,
    alpha_td=0.10,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
    eval_every=250,
    eval_episodes=50,
    final_eval_episodes=200,
    rolling_window=200,
    summary_window=1000,
    seeds=(123, 231, 777, 2024, 31415),
)

config_obj
"""

    helpers = """def media_movil(series, ventana):
    valores = np.asarray(series, dtype=float)
    kernel = np.ones(ventana, dtype=float) / ventana
    return np.convolve(valores, kernel, mode="valid")


def method_summary_df(method_results, method):
    metrics = method_results["summary"][method]
    row = {
        "Método": METHOD_LABELS[method],
        "Train reward últimos 1000": metrics["train_reward_last_window_mean"],
        "Train reward std semillas": metrics["train_reward_last_window_std"],
        "Train longitud últimos 1000": metrics["train_length_last_window_mean"],
        "Train éxito real últimos 1000": metrics["train_success_last_window_mean"],
        "Eval greedy final": metrics["final_eval_reward_mean"],
        "Eval greedy std semillas": metrics["final_eval_reward_std_across_seeds"],
        "Eval greedy longitud": metrics["final_eval_length_mean"],
        "Eval greedy éxito real": metrics["final_eval_success_rate_mean"],
    }
    return pd.DataFrame([row]).set_index("Método").round(3)


def plot_single_method(method_results, method):
    color = METHOD_COLORS[method]
    label = METHOD_LABELS[method]
    runs = method_results["runs"][method]
    curve = method_results["curves"][method]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{label} - Evolución del entrenamiento y de la política greedy", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    train_reward_curves = [media_movil(run["rewards"], config_obj.rolling_window) for run in runs]
    min_len = min(len(curva) for curva in train_reward_curves)
    matrix = np.array([curva[:min_len] for curva in train_reward_curves])
    mean_curve = matrix.mean(axis=0)
    std_curve = matrix.std(axis=0)
    x = np.arange(config_obj.rolling_window // 2, config_obj.rolling_window // 2 + min_len)
    ax1.plot(x, mean_curve, color=color, lw=2)
    ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
    ax1.set_title("Reward de entrenamiento (media móvil)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    train_length_curves = [media_movil(run["lengths"], config_obj.rolling_window) for run in runs]
    min_len = min(len(curva) for curva in train_length_curves)
    matrix = np.array([curva[:min_len] for curva in train_length_curves])
    mean_curve = matrix.mean(axis=0)
    std_curve = matrix.std(axis=0)
    x = np.arange(config_obj.rolling_window // 2, config_obj.rolling_window // 2 + min_len)
    ax2.plot(x, mean_curve, color=color, lw=2)
    ax2.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
    ax2.set_title("Longitud de episodio (media móvil)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Pasos")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 0])
    eval_x = np.array(curve["eval_episodes"])
    eval_mean = np.array(curve["eval_reward_mean"])
    eval_std = np.array(curve["eval_reward_std_across_seeds"])
    ax3.plot(eval_x, eval_mean, color=color, lw=2, marker="o", ms=3)
    ax3.fill_between(eval_x, eval_mean - eval_std, eval_mean + eval_std, color=color, alpha=0.15)
    ax3.set_title("Reward en evaluación greedy")
    ax3.set_xlabel("Episodio")
    ax3.set_ylabel("Reward medio")
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 1])
    success_mean = np.array(curve["eval_success_rate_mean"])
    success_std = np.array(curve["eval_success_rate_std_across_seeds"])
    ax4.plot(eval_x, success_mean, color=color, lw=2, marker="o", ms=3)
    ax4.fill_between(
        eval_x,
        np.clip(success_mean - success_std, 0.0, 1.0),
        np.clip(success_mean + success_std, 0.0, 1.0),
        color=color,
        alpha=0.15,
    )
    ax4.set_title("Éxito real en evaluación greedy")
    ax4.set_xlabel("Episodio")
    ax4.set_ylabel("Tasa de éxito")
    ax4.set_ylim(0, 1)
    ax4.grid(alpha=0.25)

    plt.show()


def comparative_summary_df(results):
    rows = []
    for method in results["ranking"]:
        metrics = results["summary"][method]
        rows.append(
            {
                "Método": METHOD_LABELS[method],
                "Train reward últimos 1000": metrics["train_reward_last_window_mean"],
                "Train longitud últimos 1000": metrics["train_length_last_window_mean"],
                "Train éxito real últimos 1000": metrics["train_success_last_window_mean"],
                "Eval greedy final": metrics["final_eval_reward_mean"],
                "Eval greedy std semillas": metrics["final_eval_reward_std_across_seeds"],
                "Eval greedy longitud": metrics["final_eval_length_mean"],
                "Eval greedy éxito real": metrics["final_eval_success_rate_mean"],
            }
        )
    return pd.DataFrame(rows).set_index("Método").round(3)


def plot_comparative_results(results):
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("Taxi-v3: comparación global de métodos tabulares", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    for method in results["ranking"]:
        color = METHOD_COLORS[method]
        curvas = [media_movil(run["rewards"], config_obj.rolling_window) for run in results["runs"][method]]
        min_len = min(len(curva) for curva in curvas)
        matrix = np.array([curva[:min_len] for curva in curvas])
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0)
        x = np.arange(config_obj.rolling_window // 2, config_obj.rolling_window // 2 + min_len)
        ax1.plot(x, mean_curve, lw=2, color=color, label=METHOD_LABELS[method])
        ax1.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.12)
    ax1.set_title("Reward de entrenamiento (media móvil)")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    for method in results["ranking"]:
        color = METHOD_COLORS[method]
        curvas = [media_movil(run["lengths"], config_obj.rolling_window) for run in results["runs"][method]]
        min_len = min(len(curva) for curva in curvas)
        matrix = np.array([curva[:min_len] for curva in curvas])
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0)
        x = np.arange(config_obj.rolling_window // 2, config_obj.rolling_window // 2 + min_len)
        ax2.plot(x, mean_curve, lw=2, color=color, label=METHOD_LABELS[method])
        ax2.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.12)
    ax2.set_title("Longitud de episodio (media móvil)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Pasos")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    for method in results["ranking"]:
        color = METHOD_COLORS[method]
        curve = results["curves"][method]
        x = np.array(curve["eval_episodes"])
        mean_curve = np.array(curve["eval_reward_mean"])
        std_curve = np.array(curve["eval_reward_std_across_seeds"])
        ax3.plot(x, mean_curve, lw=2, marker="o", ms=3, color=color, label=METHOD_LABELS[method])
        ax3.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.12)
    ax3.set_title("Reward en evaluación greedy")
    ax3.set_xlabel("Episodio")
    ax3.set_ylabel("Reward medio")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[1, 1])
    for method in results["ranking"]:
        color = METHOD_COLORS[method]
        curve = results["curves"][method]
        x = np.array(curve["eval_episodes"])
        mean_curve = np.array(curve["eval_success_rate_mean"])
        std_curve = np.array(curve["eval_success_rate_std_across_seeds"])
        ax4.plot(x, mean_curve, lw=2, marker="o", ms=3, color=color, label=METHOD_LABELS[method])
        ax4.fill_between(
            x,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color=color,
            alpha=0.12,
        )
    ax4.set_title("Éxito real en evaluación greedy")
    ax4.set_xlabel("Episodio")
    ax4.set_ylabel("Tasa de éxito")
    ax4.set_ylim(0, 1)
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8)

    plt.show()
"""

    on_policy_text = """## 1. Monte Carlo on-policy (every-visit)

### Qué hace este algoritmo

En Monte Carlo on-policy el agente aprende a partir de episodios completos generados por su propia política de exploración. Aquí usamos una política `epsilon-greedy` respecto a la tabla `Q`.

### Cómo se implementa aquí

La implementación sigue estos pasos:

1. Se resetea el entorno con una semilla fija para ese episodio.
2. Se genera un episodio completo usando la política `epsilon-greedy`.
3. Se almacena toda la trayectoria `(s, a, r)`.
4. Al terminar el episodio se recorre la trayectoria hacia atrás para calcular el retorno acumulado `G`.
5. Como esta versión es *every-visit*, cada aparición de `(s, a)` se actualiza con media muestral.
6. Periódicamente se evalúa la política greedy aprendida, separada de la política exploratoria.

Esta versión se eligió porque es la variante on-policy más coherente de los notebooks previos y evita el error de mezclar `first-visit` con un recorrido invertido.
"""

    run_mc_on = """mc_on_results = run_method_experiment(config_obj, "mc_on_policy", verbose=True)"""
    mc_on_summary = """method_summary_df(mc_on_results, "mc_on_policy")"""
    mc_on_plot = """plot_single_method(mc_on_results, "mc_on_policy")"""
    mc_on_commentary = method_commentary(
        METHOD_LABELS["mc_on_policy"],
        mc_on,
        "La lectura práctica es que el método mejora claramente respecto al arranque, pero sigue aprendiendo una política muy larga e ineficiente para Taxi-v3. Incluso al final de la evaluación greedy la longitud media sigue cerca de 143 pasos, lo que explica la recompensa tan negativa.",
    )

    off_policy_text = """## 2. Monte Carlo off-policy con Weighted Importance Sampling

### Qué cambia respecto al caso on-policy

Aquí ya no se aprende sobre la misma política con la que se explora.

- La política de comportamiento sigue siendo `epsilon-greedy` y es la que genera los episodios.
- La política objetivo es la política greedy respecto a `Q`.

### Cómo se implementa aquí

El procedimiento es el siguiente:

1. Se genera un episodio completo con la política de comportamiento.
2. En cada paso se guarda también `b(a|s)`, es decir, la probabilidad real de la acción bajo esa política de comportamiento.
3. Al terminar el episodio se recorre la trayectoria hacia atrás.
4. Se acumula el retorno `G` y el peso de importancia `W`.
5. Se actualiza `Q` con `Weighted Importance Sampling`.
6. Si la acción observada deja de coincidir con la política greedy objetivo, el barrido se corta.

Guardar explícitamente `b(a|s)` era una corrección importante frente a implementaciones anteriores, porque evita reconstruir esa probabilidad a posteriori con una `Q` ya modificada.
"""

    run_mc_off = """mc_off_results = run_method_experiment(config_obj, "mc_off_policy_weighted", verbose=True)"""
    mc_off_summary = """method_summary_df(mc_off_results, "mc_off_policy_weighted")"""
    mc_off_plot = """plot_single_method(mc_off_results, "mc_off_policy_weighted")"""
    mc_off_commentary = method_commentary(
        METHOD_LABELS["mc_off_policy_weighted"],
        mc_off,
        "Este método es el mejor de los dos Monte Carlo, pero su caso es especialmente interesante: la tasa de éxito greedy es alta y aún así la recompensa sigue siendo negativa. Eso indica que muchas trayectorias llegan al objetivo, pero tardando demasiado o acumulando demasiado coste por el camino.",
    )

    sarsa_text = """## 3. SARSA tabular

### Qué aprende SARSA

SARSA es un método TD on-policy. A diferencia de Monte Carlo, no espera al final del episodio para corregir la tabla `Q`, sino que actualiza en cada transición usando la acción siguiente realmente elegida por la política actual.

### Cómo se implementa aquí

En cada paso:

1. Se elige una acción `a` con la política `epsilon-greedy`.
2. Se observa `(s, a, r, s')`.
3. Si el episodio no ha terminado, se elige `a'` en `s'`.
4. Se aplica la actualización:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]`

Como es on-policy, la misma política que explora es la que se evalúa durante el aprendizaje.
"""

    run_sarsa = """sarsa_results = run_method_experiment(config_obj, "sarsa", verbose=True)"""
    sarsa_summary = """method_summary_df(sarsa_results, "sarsa")"""
    sarsa_plot = """plot_single_method(sarsa_results, "sarsa")"""
    sarsa_commentary = method_commentary(
        METHOD_LABELS["sarsa"],
        sarsa,
        "SARSA converge de forma mucho más rápida y estable que Monte Carlo en este entorno. La política greedy final ya es prácticamente óptima: 100% de éxito real y una longitud media muy cercana a 13 pasos.",
    )

    ql_text = """## 4. Q-Learning tabular

### Qué diferencia a Q-Learning de SARSA

Q-Learning es off-policy. La política de comportamiento sigue siendo `epsilon-greedy`, pero la corrección TD usa el mejor valor estimado del siguiente estado, no la acción realmente muestreada.

La actualización es:

`Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]`

### Qué esperamos ver

En Taxi-v3 este método suele ser muy fuerte porque:

- actualiza en cada paso
- propaga valores rápidamente
- y optimiza de forma más agresiva que SARSA
"""

    run_ql = """q_learning_results = run_method_experiment(config_obj, "q_learning", verbose=True)"""
    ql_summary = """method_summary_df(q_learning_results, "q_learning")"""
    ql_plot = """plot_single_method(q_learning_results, "q_learning")"""
    ql_commentary = method_commentary(
        METHOD_LABELS["q_learning"],
        ql,
        "Q-Learning queda finalmente primero, aunque por poco margen sobre SARSA. La diferencia no está en la tasa de éxito, que es del 100% en ambos casos, sino en una política ligeramente más eficiente en recompensa y longitud.",
    )

    comparison_text = """## 5. Reunimos todos los métodos y comparamos

Hasta aquí cada algoritmo se ha estudiado por separado. Ahora se combinan los cuatro resultados en una sola estructura para:

- guardar el experimento completo
- construir la tabla final agregada
- dibujar las gráficas comparativas
- y redactar una conclusión basada únicamente en los valores observados
"""

    combine_code = """all_results = combine_method_results(
    config_obj,
    {
        "mc_on_policy": mc_on_results,
        "mc_off_policy_weighted": mc_off_results,
        "sarsa": sarsa_results,
        "q_learning": q_learning_results,
    },
)
save_results(all_results, RESULTS_PATH)
print(f"Resultados guardados en: {RESULTS_PATH}")"""

    summary_code = """comparative_summary_df(all_results)"""
    ranking_code = """for idx, method in enumerate(all_results["ranking"], start=1):
    metrics = all_results["summary"][method]
    print(
        f"{idx}. {METHOD_LABELS[method]} -> "
        f"eval greedy final = {metrics['final_eval_reward_mean']:.3f} +/- "
        f"{metrics['final_eval_reward_std_across_seeds']:.3f}"
    )"""
    comparative_plot_code = """plot_comparative_results(all_results)"""

    final_commentary = f"""## Conclusiones globales

El ranking final por evaluación greedy agregada es:

{chr(10).join(ranking_lines)}

### Qué nos dicen realmente estos números

- `Q-Learning` y `SARSA` dominan el entorno con claridad. Ambos llegan al `100%` de éxito real en evaluación greedy y se mueven alrededor de `13` pasos por episodio.
- `MC Off-Policy (Weighted IS)` queda lejos en reward, aunque no en éxito. Su política greedy resuelve el problema muchas veces (`{pct(mc_off['final_eval_success_rate_mean'])}`), pero tarda demasiado (`{fmt(mc_off['final_eval_length_mean'])}` pasos de media), y por eso su reward sigue siendo negativo (`{fmt(mc_off['final_eval_reward_mean'])}`).
- `MC On-Policy (Every-Visit)` mejora bastante respecto al arranque, pero sigue siendo el peor método del conjunto: reward final `{fmt(mc_on['final_eval_reward_mean'])}` y solo `{pct(mc_on['final_eval_success_rate_mean'])}` de éxito real.

### Conclusión técnica

Para `Taxi-v3` y con este presupuesto de episodios, los métodos TD siguen siendo la opción práctica más fuerte. Monte Carlo sí aprende, y el off-policy ponderado es claramente mejor que el on-policy, pero ambos quedan por detrás cuando la comparación se hace con evaluación greedy real y bajo exactamente las mismas condiciones.

La lección metodológica más importante del experimento es que no conviene confundir:

- comportamiento durante entrenamiento
- con calidad de la política greedy aprendida

En este notebook esa diferencia queda visible sobre todo en `MC Off-Policy (Weighted IS)`.
"""

    differences_vs_v2 = """## Diferencias respecto a la versión v2

Esta implementación se diferencia de `B-MC_v2.ipynb` en varios puntos importantes:

- En `v2` se comparaban los métodos principalmente a partir de curvas de entrenamiento. En `v3` la comparación decisiva se hace con evaluación greedy separada de la exploración.
- En `v2` el éxito se interpretaba de forma indirecta. En `v3` se usa la señal real del entorno: éxito solo cuando el episodio termina por `terminated`.
- En `v2` Monte Carlo off-policy no dejaba explicitada de forma robusta la probabilidad `b(a|s)` usada en importance sampling. En `v3` esa probabilidad se guarda en cada transición y se reutiliza al actualizar.
- En `v2` no había una igualdad experimental tan explícita entre algoritmos. En `v3` todos comparten exactamente las mismas semillas, episodios, `gamma`, planificación de `epsilon`, frecuencia de evaluación y criterio de resumen.
- En `v2` la narrativa final se apoyaba sobre todo en la inspección visual de las gráficas. En `v3` cada afirmación cuantitativa se apoya en métricas agregadas sobre varias semillas y en una tabla final comparativa.
- En `v2` el notebook estaba centrado solo en Monte Carlo. En `v3` primero se valida Monte Carlo y después se compara de forma justa con `SARSA` y `Q-Learning` bajo el mismo protocolo.
- En `v3` también se añaden gráficas parciales por algoritmo para entender su dinámica antes de pasar a la comparación global.

En resumen, `v2` era útil como cuaderno exploratorio y descriptivo, mientras que `v3` está planteado como un experimento reproducible y comparable, con conclusiones apoyadas en evaluación objetiva.
"""

    notebook = {
        "cells": [
            md_cell(intro),
            md_cell(environment),
            md_cell(protocol),
            md_cell(module_note),
            code_cell(imports),
            code_cell(helpers),
            md_cell(on_policy_text),
            code_cell(run_mc_on),
            code_cell(mc_on_summary),
            code_cell(mc_on_plot),
            md_cell(mc_on_commentary),
            md_cell(off_policy_text),
            code_cell(run_mc_off),
            code_cell(mc_off_summary),
            code_cell(mc_off_plot),
            md_cell(mc_off_commentary),
            md_cell(sarsa_text),
            code_cell(run_sarsa),
            code_cell(sarsa_summary),
            code_cell(sarsa_plot),
            md_cell(sarsa_commentary),
            md_cell(ql_text),
            code_cell(run_ql),
            code_cell(ql_summary),
            code_cell(ql_plot),
            md_cell(ql_commentary),
            md_cell(comparison_text),
            code_cell(combine_code),
            code_cell(summary_code),
            code_cell(ranking_code),
            code_cell(comparative_plot_code),
            md_cell(final_commentary),
            md_cell(differences_vs_v2),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main() -> None:
    with RESULTS_PATH.open("r", encoding="utf-8") as fh:
        results = json.load(fh)

    notebook = build_notebook(results)
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as fh:
        json.dump(notebook, fh, indent=2, ensure_ascii=False)
    print(f"Notebook generado en: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
