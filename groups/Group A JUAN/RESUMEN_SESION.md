# Resumen de sesion — Agente Juan (FlatMonteCarlo)

## Que hicimos

### 1. Script de entrenamiento de Santi vs Juan (`train_against_juan.py`)
Lo primero que hicimos fue un script para que Santi pueda entrenar su politica (FVMC / Q-table)
progresivamente contra el agente de Juan. El script:
- Carga la Q-table de Santi o la entrena desde cero si no existe.
- Entrena por batches de 1,000 episodios alternando colores.
- Despues de cada batch, mide el win % de Santi en 10 partidas de prueba.
- Sube la dificultad de Juan (N=200 → 500 → 1000 → 2000) cuando Santi alcanza 70% de victorias.
- Guarda progreso en `training_progress.json` y genera `training_progress.png`.
- Se puede interrumpir y reanudar con Ctrl+C.

Adicionalmente se creo `INSTRUCCIONES_SANTI.md` con la guia paso a paso para que Santi lo corra solo.

---

### 2. Notebook de analisis de Juan (`groups/Group A JUAN/entrega.ipynb`)
Se creo el notebook requerido por el reto con 4 experimentos sobre el agente FlatMonteCarlo.
Todos los experimentos analizan el **mismo algoritmo actual**, variando el parametro N.

#### Experimentos

| # | Nombre | Que mide | Variable |
|---|--------|----------|----------|
| 1 | Win rate vs Random | Calidad del agente contra oponente aleatorio | N (10–200) |
| 2 | Trade-off tiempo/calidad | Tiempo de decision por movimiento | N (10–1000) |
| 3 | Auto-desempeno | Win rate FlatMC(N=a) vs FlatMC(N=b) — heatmap | N de ambos |
| 4 | Ventaja de color | Diferencia rojo vs amarillo con y sin oponente | N simetrico |

**V1 = N=100** (version rapida), **V2 = N=200** (version final entregada) — etiquetados en las graficas
con lineas punteadas (Exp 1/2) y rectangulos resaltados (Exp 3).

---

## Preguntas que surgieron y sus respuestas

### "¿Cual es la diferencia entre el Experimento 1 y el Experimento 2?"

Son dos experimentos sobre el **mismo codigo actual**, con dimensiones distintas:
- **Exp 1** = eje de CALIDAD: ¿cuanto gana el agente en funcion de N?
- **Exp 2** = eje de COSTO: ¿cuanto tiempo tarda cada decision en funcion de N?

Ninguno es una propuesta de mejora. La propuesta de mejora esta en la **seccion 6** del notebook.

### "¿El Experimento 2 es una propuesta de mejora?"

No. El Exp 2 usa exactamente el mismo `FlatMonteCarlo` y mide cuanto tarda `act()` 
en distintos valores de N. Lo que revela es el cuello de botella (rollouts uniformes), 
que luego **motiva** la propuesta de MCTS con UCB1 en la seccion 6.

### "¿Que son V1 y V2 en las graficas?"

Son las dos "versiones" del agente que la rubrica exige comparar (criterio 2, nivel 100%):
- **V1 = N=100**: version rapida, menos precisa
- **V2 = N=200**: la version final entregada al torneo

Aparecen como lineas verticales punteadas en Exp 1 y 2, y como rectangulos en el heatmap del Exp 3.

---

## Hallazgos clave de cada grafica

### Exp 1 — Win rate vs Random
- Tanto rojo como amarillo alcanzan ~100% de win rate **desde N=10**.
- Solo hay una ganancia visible entre N=10 y N=25 (~5%); despues esta saturado.
- V1 y V2 son **indistinguibles** contra el aleatorio. Para ver diferencias necesitas Exp 3.

### Exp 2 — Tiempo de computo
- Crecimiento perfectamente **lineal** con N (confirmado por ajuste lineal).
- V1 (N=100) ≈ 300 ms/decision | V2 (N=200) ≈ 680 ms/decision.
- Para N ≤ 200: mas de 85 decisiones posibles en 1 minuto — no hay problema de budget.
- N=1000 ya no alcanza a terminar una partida completa (~17 decisiones vs 21 necesarias).

### Exp 3 — Auto-desempeno (heatmap)
- Clara dominancia por N: V2 gana 100% vs N=10 y N=50, 60% vs V1 (N=100).
- V1 gana solo el 30% contra V2.
- Win rate promedio: V2 ≈ 77% | V1 ≈ 62% — diferencia de 15 puntos.
- La diagonal del heatmap (igual N) muestra resultados variables (20-70%) por muestra pequeña.

### Exp 4 — Ventaja de color
- Ambos colores dominan al random igual de bien desde N=25.
- Con N simetrico: los resultados son inconsistentes (amarillo gana mas a N=10/50, rojo a N=100, empate a N=200).
- Conclusion: muestra insuficiente para afirmar ventaja de primer jugador. N=200 da 50/50.

---

## Archivos generados

```
groups/Group A JUAN/
  policy.py              <- codigo del agente (sin tocar)
  entrega.ipynb          <- notebook de analisis (NUEVO)
  RESUMEN_SESION.md      <- este archivo
  ENTREGA_PDF.md         <- documento listo para convertir a PDF
  _exp_cache.json        <- cache de resultados (no subir a git)
  exp1_vs_random.png
  exp2_timing.png
  exp3_selfplay.png
  exp4_color_bias.png
  resumen_analisis.png

train_against_juan.py    <- script entrenamiento Santi vs Juan (raiz)
INSTRUCCIONES_SANTI.md   <- guia para Santi (raiz)
```
