# Aprendizaje por Refuerzo: Bandido de k--brazos y Entornos Complejos

## Información académica

- **Autores:** Aida García Echevarría, Christian Andrés Rueda Ayala, Pablo Daniel Cuña Cabrera
- **Asignatura:** Aprendizaje por Refuerzo
- **Curso:** 2025/2026

## Resumen del proyecto

Este repositorio reúne varios trabajos prácticos desarrollados en el marco de la asignatura **Aprendizaje por Refuerzo**. El contenido se organiza en dos bloques principales:

1. **Problemas de bandido de k brazos**, centrados en el equilibrio entre exploración y explotación en entornos estocásticos simples.
2. **Entornos complejos**, donde se estudian métodos de control por refuerzo en entornos discretos y continuos mediante enfoques Monte Carlo, métodos tabulares TD y aproximación funcional.

El objetivo general del proyecto es comparar algoritmos de aprendizaje por refuerzo bajo protocolos experimentales explícitos, con especial atención a:

- la calidad final de la política aprendida,
- la velocidad de aprendizaje,
- la estabilidad entre ejecuciones,
- y la relación entre decisiones de modelado, representación del estado y comportamiento empírico.

## Organización del repositorio

```text
extML/
├── README.md
├── requirements.txt
├── k_brazos/
│   ├── main.ipynb
│   ├── banditGreedy.ipynb
│   ├── banditSoftmax.ipynb
│   ├── banditUCB.ipynb
│   └── src/
│       ├── algorithms/
│       ├── arms/
│       └── plotting/
└── entornos_complejos/
    ├── A_MonteCarloTodasLasVisitas.ipynb
    ├── B_metodos_tabulares.ipynb
    ├── B_control_aprox.ipynb
    ├── main.ipynb
    └── src/
        ├── agent.py
        ├── tabular_taxi.py
        ├── control_aprox_utils.py
        ├── features.py
        ├── sarsa_agent.py
        ├── dqn_agent.py
        ├── training.py
        └── __init__.py
```

## Bloque 1: `k_brazos`

### Propósito

Este bloque estudia el problema clásico del **bandido de k brazos**, donde el agente debe decidir entre explorar acciones inciertas o explotar la mejor estimación disponible. Se analizan varias políticas de selección de brazos sobre distintas distribuciones de recompensa.

### Algoritmos implementados

- **Epsilon-greedy**
- **Softmax (Boltzmann sobre valores estimados)**
- **UCB1**
- **UCB2**

### Distribuciones de recompensa modeladas

- **Normal**
- **Bernoulli**
- **Binomial**

### Notebooks principales

- [`k_brazos/banditGreedy.ipynb`](/home/christianr/extML/k_brazos/banditGreedy.ipynb)  
  Estudio específico de configuraciones de epsilon-greedy.

- [`k_brazos/banditSoftmax.ipynb`](/home/christianr/extML/k_brazos/banditSoftmax.ipynb)  
  Estudio específico del método Softmax.

- [`k_brazos/banditUCB.ipynb`](/home/christianr/extML/k_brazos/banditUCB.ipynb)  
  Comparación de variantes UCB1 y UCB2 con diferentes parámetros.

- [`k_brazos/main.ipynb`](/home/christianr/extML/k_brazos/main.ipynb)  
  Comparación integrada de las mejores configuraciones de epsilon-greedy, Softmax y UCB.

### Código reutilizable

- [`k_brazos/src/algorithms/`](/home/christianr/extML/k_brazos/src/algorithms)  
  Implementa la jerarquía de algoritmos de selección de brazos.

- [`k_brazos/src/arms/`](/home/christianr/extML/k_brazos/src/arms)  
  Define los tipos de brazos y el objeto `Bandit`.

- [`k_brazos/src/plotting/`](/home/christianr/extML/k_brazos/src/plotting)  
  Centraliza las utilidades de visualización de recompensa media, selección del brazo óptimo, regret y estadísticas por brazo.

### Métricas utilizadas

Los notebooks de este bloque comparan políticas con base en:

- recompensa promedio,
- porcentaje de selección del brazo óptimo,
- regret acumulado,
- y estadísticas de selección por brazo.

## Bloque 2: `entornos_complejos`

### Propósito

Este bloque extiende el estudio hacia problemas de control más ricos, incluyendo entornos discretos y continuos. Se consideran tanto métodos sin modelo como métodos basados en aproximación funcional.

### Estudios incluidos

#### 1. Monte Carlo con políticas epsilon-soft en FrozenLake

- [`entornos_complejos/A_MonteCarloTodasLasVisitas.ipynb`](/home/christianr/extML/entornos_complejos/A_MonteCarloTodasLasVisitas.ipynb)

Este notebook presenta un experimento introductorio con **Monte Carlo every-visit** sobre **FrozenLake-v1** en versiones `4x4` y `8x8`, ambas configuradas sin resbalamiento (`is_slippery=False`) para facilitar la interpretación inicial del aprendizaje.

#### 2. Comparación de métodos tabulares en Taxi-v3

- [`entornos_complejos/B_metodos_tabulares.ipynb`](/home/christianr/extML/entornos_complejos/B_metodos_tabulares.ipynb)

Este notebook compara, bajo un protocolo homogéneo, los siguientes algoritmos:

- Monte Carlo on-policy (every-visit),
- Monte Carlo off-policy con **Weighted Importance Sampling**,
- **SARSA** tabular,
- **Q-Learning** tabular.

El diseño experimental se apoya en un protocolo de semillas, ventanas de resumen y evaluación greedy consistente entre métodos, con análisis cuantitativo y conclusiones fundamentadas sobre el entorno **Taxi-v3**.

El núcleo reusable de este estudio está en [`entornos_complejos/src/tabular_taxi.py`](/home/christianr/extML/entornos_complejos/src/tabular_taxi.py).

#### 3. Control aproximado en CartPole-v1 y Tetris

- [`entornos_complejos/B_control_aprox.ipynb`](/home/christianr/extML/entornos_complejos/B_control_aprox.ipynb)

Este notebook estudia control aproximado en dos entornos de distinta naturaleza:

- **CartPole-v1**, como benchmark continuo y controlado,
- **Tetris**, como problema con mayor complejidad estructural.

Los algoritmos comparados son:

- **SARSA semi-gradiente** con aproximador lineal,
- **Deep Q-Network (DQN)** con red neuronal multicapa.

Una decisión metodológica central del notebook es que ambos métodos comparten:

- la misma representación de entrada por entorno,
- el mismo calendario de exploración por episodio,
- las mismas semillas de entrenamiento y evaluación,
- y la misma convención estadística para resumir resultados.

Con ello, la comparación puede interpretarse en términos algorítmicos y no como consecuencia de codificaciones distintas del estado.

El código reusable de este bloque está centralizado en [`entornos_complejos/src/control_aprox_utils.py`](/home/christianr/extML/entornos_complejos/src/control_aprox_utils.py).  
Los módulos [`features.py`](/home/christianr/extML/entornos_complejos/src/features.py), [`sarsa_agent.py`](/home/christianr/extML/entornos_complejos/src/sarsa_agent.py), [`dqn_agent.py`](/home/christianr/extML/entornos_complejos/src/dqn_agent.py) y [`training.py`](/home/christianr/extML/entornos_complejos/src/training.py) se mantienen como capas de compatibilidad sobre dicha implementación consolidada.

## Dependencias

### Dependencias base

El archivo [`requirements.txt`](/home/christianr/extML/requirements.txt) incluye actualmente dependencias de visualización y cálculo empleadas en el bloque de bandits:

- `numpy`
- `seaborn`
- `matplotlib`

### Dependencias adicionales para `entornos_complejos`

Para ejecutar los notebooks de entornos complejos se requieren además:

- `gymnasium[classic-control]`
- `tetris-gymnasium`
- `torch`
- `pandas`
- `ipython`
- `imageio` de forma opcional, solo para exportación de GIFs

## Instalación recomendada

Se recomienda trabajar en un entorno virtual aislado.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "gymnasium[classic-control]>=0.29" "tetris-gymnasium" "torch>=2.0" "pandas>=2.0" ipython imageio notebook
```

Si se utiliza JupyterLab en lugar de `notebook`, puede sustituirse el último paquete por `jupyterlab`.

## Ejecución recomendada

### Bloque `k_brazos`

1. Ejecutar primero los notebooks específicos si se desea estudiar cada familia de algoritmos por separado:
   - `banditGreedy.ipynb`
   - `banditSoftmax.ipynb`
   - `banditUCB.ipynb`
2. Ejecutar después [`k_brazos/main.ipynb`](/home/christianr/extML/k_brazos/main.ipynb) para una comparación agregada de las mejores configuraciones.

### Bloque `entornos_complejos`

1. Ejecutar [`A_MonteCarloTodasLasVisitas.ipynb`](/home/christianr/extML/entornos_complejos/A_MonteCarloTodasLasVisitas.ipynb) para el estudio introductorio de Monte Carlo en FrozenLake.
2. Ejecutar [`B_metodos_tabulares.ipynb`](/home/christianr/extML/entornos_complejos/B_metodos_tabulares.ipynb) para la comparación objetiva en Taxi-v3.
3. Ejecutar [`B_control_aprox.ipynb`](/home/christianr/extML/entornos_complejos/B_control_aprox.ipynb) para el estudio de control aproximado en CartPole-v1 y Tetris.

### Modos de ejecución en `B_control_aprox.ipynb`

El notebook de control aproximado distingue entre dos modos:

- `FAST`: útil para validar el pipeline, imports y coherencia general.
- `FULL`: pensado para producir resultados finales y análisis comparativos completos.

## Criterios metodológicos destacados

El repositorio no reúne únicamente implementaciones, sino estudios comparativos donde la calidad metodológica es parte central del trabajo. Entre los criterios empleados destacan:

- uso de semillas explícitas,
- separación entre rendimiento de entrenamiento y evaluación greedy,
- resúmenes por ventana final y por semilla,
- comparaciones entre algoritmos bajo representación común cuando procede,
- y análisis de resultados redactados a partir de métricas observadas y no de impresiones visuales aisladas.

## Autoría y procedencia del código

Parte del código base del bloque `k_brazos` y algunos materiales de apoyo proceden de plantillas docentes proporcionadas en la asignatura y referenciadas en los encabezados de los propios archivos. Sobre esa base se han realizado ampliaciones, correcciones y desarrollos adicionales por parte del equipo autor del proyecto.

En el bloque `entornos_complejos`, el trabajo combina desarrollo propio, refactorización progresiva del código experimental y consolidación de utilidades reutilizables para hacer comparables los estudios tabulares y aproximados.

## Observaciones finales

- El repositorio está orientado principalmente a un flujo de trabajo basado en notebooks.
- Los notebooks contienen tanto implementación como análisis de resultados.
- [`entornos_complejos/main.ipynb`](/home/christianr/extML/entornos_complejos/main.ipynb) está presente en el repositorio, pero no actúa actualmente como notebook integrador del bloque.

Este repositorio debe leerse, por tanto, como una colección estructurada de experimentos académicos sobre exploración, aprendizaje tabular y control aproximado en aprendizaje por refuerzo.
