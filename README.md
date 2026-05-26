# Connect-4-IA — Grupo C (SANTI) — FVMC

Repositorio del proyecto de Connect 4. Incluye el agente, los experimentos y el notebook de entrega con graficas y validacion.

## Uso

### Generar resultados completos
```
python generate_data.py
```
Genera `results.json` con todos los experimentos.

### Generar solo comparaciones directas
```
python run_head_to_head.py
```
Genera `results_h2h.json`. Puedes ajustar el numero de partidas en `N_GAMES` y el presupuesto de rollouts con `FLATMC_N`.

### Notebook de entrega
Abrir `entrega.ipynb` y ejecutar:
- Celda 1 para cargar `results.json` (experimentos completos), o
- Celda 2 para cargar `results_h2h.json` (comparaciones directas).

Las graficas se guardan en `graficas/`.

## Como funciona
El agente principal usa Monte Carlo con tabla Q y heuristica ligera para balancear exploracion y decision informada. Los experimentos cubren:
- Aprendizaje vs aleatorio y desempeno por color.
- Sensibilidad al parametro `heuristic_w`.
- Self-play para consistencia interna.
- Cobertura de la tabla Q y relacion con el presupuesto de entrenamiento.
- Comparaciones directas con agentes de otros integrantes.

## Dependencias
- Python 3.10+
- Paquetes: `numpy`, `matplotlib`

Instalacion rapida:
```
python -m pip install numpy matplotlib
```

## Estructura
- `groups/`: agentes por integrante.
- `connect4/`: motor y utilidades del juego.
- `entrega.ipynb`: notebook de analisis y graficas.
- `generate_data.py`: genera `results.json`.
- `run_head_to_head.py`: genera `results_h2h.json`.
- `tournament.py`, `run_games.py`: utilidades para correr partidas.

## Notas
- `generate_data.py` puede tardar varios minutos por los entrenamientos.
- Para ejecuciones rapidas, reduce `N_GAMES` y `FLATMC_N` en `run_head_to_head.py`.
