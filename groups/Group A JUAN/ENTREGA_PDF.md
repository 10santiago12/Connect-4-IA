# Agente Connect-4: Flat Monte Carlo
**Juan Gomez — Group A | Fundamentos de IA 2026.1**

---

## 1. Idea principal y diferencia respecto al grupo

El agente implementado es **Flat Monte Carlo (FMC)**: dado el estado actual del tablero,
estima el valor de cada columna legal ejecutando `N` simulaciones aleatorias (*rollouts*)
desde esa jugada hasta el final de la partida. Elige la columna con mayor tasa de victoria estimada.

**Lo que lo diferencia de los otros agentes del grupo:**

- Es completamente **online**: no tiene fase de entrenamiento ni memoria entre partidas.
  Toda la inteligencia emerge en tiempo de decision.
- Su unico parametro es `N` (rollouts por accion), lo que lo hace totalmente transparente
  y permite analizar su comportamiento como funcion de un solo tornillo numerico.
- Escala naturalmente con el tiempo disponible: mas budget = mayor N = mejor calidad,
  sin necesidad de reentrenar ni guardar nada.

**Codigo final (branch `Politica-Juan`):**
`https://github.com/10santiago12/Connect-4-IA/blob/Politica-Juan/groups/Group%20A%20JUAN/policy.py`

---

## 2. Versiones evaluadas

| Version | N   | Descripcion |
|---------|-----|-------------|
| **V1**  | 100 | Rapida — decision en ~300 ms |
| **V2**  | 200 | **Version final entregada** — decision en ~680 ms, mejor calidad vs oponentes fuertes |

---

## 3. Analisis del agente

El analisis completo esta en `entrega.ipynb`. A continuacion se presentan las conclusiones
de los cuatro experimentos ejecutados.

### 3a. Configuracion del agente: efecto de N

**Exp 1 — Win rate vs jugador aleatorio** (N ∈ {10, 25, 50, 100, 200}, 20 partidas por N):
FlatMC alcanza ~100% de win rate contra el aleatorio **desde N=10**, tanto como rojo como amarillo.
V1 y V2 son indistinguibles contra este oponente. Conclusion: el agente supera ampliamente
el prerequisito de la rubrica (>50%) incluso con recursos minimos.

**Exp 2 — Trade-off tiempo / calidad** (N ∈ {10, …, 1000}):
El tiempo de decision crece linealmente con N (R² ≈ 1). V2 (680 ms/decision) permite
mas de 85 movimientos dentro del budget de 1 minuto — sin restriccion practica.
Solo a partir de N≈500 el budget empieza a ser un limitante real.

**Cuello de botella:** FlatMC distribuye N rollouts *uniformemente* entre todas las columnas
legales, incluyendo jugadas claramente malas. Con 5–7 columnas en mid-game, la mayoria
del presupuesto se desperdicia.

### 3b. Oponente: el agente contra si mismo

**Exp 3 — Auto-desempeno, heatmap N_a vs N_b** (10 partidas por par, ambos colores):
Mayor N domina consistentemente. V2 (N=200) gana **100% contra N=10 y N=50**, y **60% contra V1 (N=100)**.
V1 solo gana el 30% contra V2. En terminos de win rate promedio vs todos los oponentes:
V2 ≈ **77%** frente al **62%** de V1 — 15 puntos de diferencia — justificando empiricamente
la eleccion de N=200 como version final.

**Exp 4 — Ventaja de color** (diagonal del heatmap + datos del Exp 1):
Ambos colores dominan al aleatorio de manera identica desde N=25.
Cuando ambos agentes tienen el mismo N, los resultados son inconsistentes por muestra pequena
(10 partidas). A N=200 simetrico el resultado es 50/50, confirmando que el desempeno del
agente depende de N y no del color asignado.

---

## 4. Propuesta de mejora

**Problema:** rollouts uniformes entre columnas es ineficiente — el presupuesto N
se gasta en jugadas malas igual que en jugadas buenas.

**Mejora propuesta: MCTS con UCB1.**
En lugar de distribuir N rollouts equitativamente, UCB1 guia la exploracion hacia las
columnas mas prometedoras mediante:

> UCB1(s, a) = q_hat(s,a) + C · sqrt(ln N_parent / N_a)

El segundo termino es un bonus de exploracion que disminuye a medida que una accion
se visita mas. Esto concentra el mismo budget N donde realmente importa.

**Por que resuelve el cuello de botella:**
Con el mismo N=200, MCTS-UCB1 deberia ganarle a FlatMC(N=200) en auto-desempeno y
alcanzar calidad equivalente a FlatMC(N=500) en menos tiempo — prediccion verificable
ejecutando los mismos Exp 1–3 con el nuevo agente.

**Mejora secundaria de bajo costo:** hacer que dentro del rollout, si un jugador puede
ganar en un movimiento lo haga. Rollouts mas realistas = estimaciones mas precisas, sin aumentar N.
