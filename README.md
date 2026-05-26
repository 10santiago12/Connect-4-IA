# QLearningAgent — Connect-4

**Autor:** Santiago · Fundamentos de IA · Universidad de La Sabana · 2026-1  
**Repositorio:** https://github.com/10santiago12/Connect-4-IA/tree/guti

---

## Idea del agente

Agente basado en **Q-Learning tabular**. Aprende la función de valor de acción `q(s, a)` actualizando en cada paso con bootstrapping:

```
q(s,a) ← q(s,a) + α · [r + γ · max_a' q(s',a') − q(s,a)]
```

A diferencia de Monte Carlo (que espera al final del episodio), Q-Learning actualiza en cada transición, lo que acelera la convergencia.

---

## Archivos

| Archivo | Descripción |
|---|---|
| `policy.py` | Código completo del agente |
| `qlearning_table.pkl` | Tabla Q pre-entrenada (carga automática en `mount()`) |
| `README.md` | Este archivo |

---

## Cómo ejecutar

### 1. Desde `run_games.py` (interfaz gráfica)

```bash
python run_games.py
```

Selecciona **Group C SANTI** como uno de los jugadores y presiona **Iniciar**.

### 2. Desde `main.py` (torneo)

```bash
python main.py
```

El agente se detecta automáticamente junto con los demás del grupo.

---

## Comportamiento de `mount()`

- **Primera ejecución:** si no existe `qlearning_table.pkl`, entrena desde cero (80,000 episodios vs aleatorio + 30,000 self-play) y guarda la tabla en disco.
- **Ejecuciones siguientes:** carga la tabla directamente desde `qlearning_table.pkl` en milisegundos.

> Se recomienda incluir `qlearning_table.pkl` en el repositorio para evitar el entrenamiento inicial.

---

## Hiperparámetros

| Parámetro | Valor | Descripción |
|---|---|---|
| `ALPHA` | 0.25 | Tasa de aprendizaje |
| `GAMMA` | 0.95 | Factor de descuento |
| `EPSILON` | 0.15 | Probabilidad de exploración durante entrenamiento |
| `N_VS_RANDOM` | 80,000 | Episodios de entrenamiento contra jugador aleatorio |
| `N_SELF_PLAY` | 30,000 | Episodios de self-play |

---

## Lógica de decisión (`act`)

En cada turno el agente aplica 4 reglas en orden de prioridad:

1. **Ganar ya** — si existe una columna que da victoria inmediata, la juega.
2. **Bloquear** — si el oponente puede ganar en su siguiente turno, bloquea esa columna.
3. **No regalar victoria** — descarta columnas que le abren una victoria al oponente en el turno siguiente (lookahead 3 jugadas).
4. **Tabla Q** — sobre las columnas restantes, elige la de mayor valor `q(s, a)`.