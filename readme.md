# Agente Flat Monte Carlo — Grupo A (Juan)

## Uso

El agente se descubre y corre automaticamente con el torneo. No requiere instalacion ni entrenamiento previo.

```bash
python main.py
```

## Como funciona

Para cada decision, el agente ejecuta `N` simulaciones aleatorias (*rollouts*) por cada columna legal y elige la que tenga mayor tasa de victoria estimada. No guarda nada entre partidas.

El unico parametro es `N` (por defecto `N=200`). Para cambiarlo, editar la linea en `policy.py`:

```python
class FlatMonteCarlo(Policy):
    def __init__(self, N: int = 200):  # <-- cambiar aqui
```

## Dependencias

Solo las del proyecto base: `numpy`. Nada adicional.
