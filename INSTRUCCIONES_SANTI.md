# Instrucciones: Entrenamiento progresivo de tu policy (Santi) vs Juan

## Qué hace este proceso

El script `train_against_juan.py` entrena tu política (FVMC / Q-table) de forma progresiva
contra la política de Juan (Flat Monte Carlo). El objetivo es ver qué tan bien aprende
tu Q-table a ganarle a Juan a medida que aumentas los episodios de entrenamiento y,
eventualmente, al aumentar la dificultad de Juan (su parámetro N).

### Flujo del script

1. Carga tu Q-table actual (`fvmc_qtable.pkl`) o la entrena desde cero si no existe.
2. Entrena tu política contra Juan en batches de 1,000 episodios.
3. Después de cada batch, juega 10 partidas de prueba y mide tu win %.
4. Guarda el progreso en `training_progress.json` y actualiza `fvmc_qtable.pkl`.
5. Cuando alcanzas 70% de victorias, sube la dificultad de Juan (N más alto).
6. Al terminar (o al interrumpir con Ctrl+C), genera la gráfica `training_progress.png`.

---

## Pasos para correr el entrenamiento

### 1. Actualizar el repositorio

```bash
git pull
```

### 2. Verificar dependencias

```bash
pip install numpy matplotlib pydantic
```

### 3. (Opcional) Empezar desde cero

Si quieres ignorar cualquier Q-table previa y empezar el entrenamiento desde cero,
borra el archivo de caché antes de correr el script:

- **Windows:** `del fvmc_qtable.pkl`
- **Mac/Linux:** `rm fvmc_qtable.pkl`

También puedes borrar el progreso anterior:

- **Windows:** `del training_progress.json`
- **Mac/Linux:** `rm training_progress.json`

### 4. Correr el script

```bash
python train_against_juan.py
```

Verás en consola algo como:

```
============================================================
  Entrenamiento progresivo: Santi vs Juan
============================================================

Cargando Q-table de Santi desde 'fvmc_qtable.pkl'...
  Estados conocidos: 142,387

────────────────────────────────────────────────────────────
  Nivel: Juan N=200
  Total episodios hasta ahora: 0
────────────────────────────────────────────────────────────
  Batch  1/50 | Episodios:   1,000 | Win %:  30.0% | Empates: 1/10 | Estados Q: 145,201
  Batch  2/50 | Episodios:   2,000 | Win %:  40.0% | Empates: 0/10 | Estados Q: 148,344
  ...
  ✓ Target alcanzado (70%) con Juan N=200!

  Subiendo dificultad: Juan N=200 → N=500
  ...
```

### 5. Interrumpir y continuar

Puedes interrumpir en cualquier momento con **Ctrl+C**. El script:
- Guarda la Q-table actualizada en `fvmc_qtable.pkl`.
- Guarda el progreso en `training_progress.json`.
- Genera la gráfica `training_progress.png`.

Para **continuar donde quedaste**, simplemente vuelve a correr:

```bash
python train_against_juan.py
```

El script detecta automáticamente el progreso previo y retoma desde ahí.

---

## Archivos generados

| Archivo | Descripción |
|---|---|
| `fvmc_qtable.pkl` | Q-table de tu política (actualizada en cada batch) |
| `training_progress.json` | Historial de checkpoints con win % por episodio |
| `training_progress.png` | Gráfica del progreso de entrenamiento |

---

## Interpretar la gráfica

La gráfica `training_progress.png` muestra:

- **Eje X:** Episodios de entrenamiento acumulados.
- **Eje Y:** Win % de tu política en las 10 partidas de prueba.
- **Línea verde punteada:** Target de 70% de victorias.
- **Zonas sombreadas:** Cada zona de color es un nivel de N de Juan
  (azul = N=200, naranja = N=500, rojo = N=1000, verde azulado = N=2000).

Lo ideal es ver la línea de win % subir con más entrenamiento y cruzar el target.

---

## Parámetros configurables

Si quieres ajustar el comportamiento, edita las constantes al inicio de `train_against_juan.py`:

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `JUAN_N_LEVELS` | `[200, 500, 1000, 2000]` | Niveles de N de Juan |
| `BATCH_SIZE` | `1000` | Episodios de entrenamiento por batch |
| `TEST_GAMES` | `10` | Partidas de prueba por checkpoint |
| `WIN_TARGET` | `0.70` | Win % objetivo para subir de nivel |
| `MAX_BATCHES` | `50` | Máximo de batches por nivel de N |

---

## Nota sobre tiempo de ejecución

- Cada batch de 1,000 episodios tarda aproximadamente **1-3 minutos** dependiendo de tu máquina.
- El entrenamiento completo (todos los niveles de N) puede tomar varias horas.
- Puedes interrumpir y continuar en cualquier momento sin perder progreso.
- El tiempo de Juan por movimiento aumenta con N — con N=2000 cada partida de prueba
  puede tomar hasta 30 segundos.
