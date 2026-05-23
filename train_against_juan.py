"""
Script de entrenamiento progresivo: Santi (FVMCPolicy) vs Juan (FlatMonteCarlo).

Flujo:
  1. Carga o crea la Q-table de Santi.
  2. Si es nueva, corre el entrenamiento base de Santi (vs random + self-play).
  3. Bucle por niveles de N de Juan [200, 500, 1000, 2000]:
       - Entrena 1000 episodios de Santi explorando contra Juan.
       - Mide win % en 10 partidas de prueba.
       - Guarda checkpoint en training_progress.json.
       - Repite hasta alcanzar 70% de victorias o 50 batches.
       - Al alcanzar el target, sube el N de Juan.
  4. Genera training_progress.png al terminar o al interrumpir con Ctrl+C.

Uso:
  python train_against_juan.py
"""

import sys
import os
import json
import pickle
import signal
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Asegurar que los imports del proyecto funcionen (los folders tienen espacios)
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from connect4.connect_state import ConnectState

# Importar con importlib porque los nombres de carpeta tienen espacios
FVMCPolicy = importlib.import_module("groups.Group C SANTI.policy").FVMCPolicy
FlatMonteCarlo = importlib.import_module("groups.Group A JUAN.policy").FlatMonteCarlo

# ── Configuración ─────────────────────────────────────────────────────────────

JUAN_N_LEVELS   = [200, 500, 1000, 2000]   # Niveles de N de Juan
BATCH_SIZE      = 1_000                    # Episodios de entrenamiento por batch
TEST_GAMES      = 10                       # Partidas de prueba por checkpoint
WIN_TARGET      = 0.70                     # Win % objetivo para subir de nivel
MAX_BATCHES     = 50                       # Máximo de batches por nivel de N

SANTI_CACHE     = "fvmc_qtable.pkl"
PROGRESS_FILE   = "training_progress.json"
GRAPH_FILE      = "training_progress.png"

# ── Internals de Santi (re-exportados como funciones module-level) ─────────────

def _pack(board_pov: np.ndarray) -> bytes:
    """Codifica tablero 6×7 en 11 bytes (2 bits/celda). Replica la de Santi."""
    out = 0
    for v in (board_pov.ravel() + 1).astype(np.uint8):
        out = (out << 2) | int(v)
    return out.to_bytes(11, 'big')


# ── Simulación de un juego ────────────────────────────────────────────────────

def run_game_train(santi: FVMCPolicy, juan: FlatMonteCarlo, santi_color: int) -> int:
    """
    Corre un episodio de entrenamiento. Santi explora con _explore(), Juan juega greedy.
    Actualiza la Q-table de Santi al finalizar.
    Retorna el ganador (-1 o 1, 0 = empate).
    """
    state = ConnectState()
    traj = []
    first_santi = True

    while not state.is_final():
        free = state.get_free_cols()
        if state.player == santi_color:
            bpov = state.board * santi_color
            if first_santi:
                col = int(np.random.choice(free))
                first_santi = False
            else:
                col = santi._explore(bpov, free)
            traj.append((_pack(bpov), col))
        else:
            col = juan.act(state.board)
        state = state.transition(col)

    winner = state.get_winner()
    G = 1.0 if winner == santi_color else (0.0 if winner == 0 else -1.0)
    santi._update(traj, G)
    return winner


def run_game_test(santi: FVMCPolicy, juan: FlatMonteCarlo, santi_color: int) -> int:
    """
    Corre una partida de prueba. Santi juega greedy (act()), Juan juega greedy.
    Retorna el ganador (-1 o 1, 0 = empate).
    """
    state = ConnectState()
    while not state.is_final():
        if state.player == santi_color:
            col = santi.act(state.board)
        else:
            col = juan.act(state.board)
        state = state.transition(col)
    return state.get_winner()


# ── Test: medir win % de Santi en TEST_GAMES partidas ────────────────────────

def evaluate(santi: FVMCPolicy, juan: FlatMonteCarlo) -> tuple[float, int]:
    """Juega TEST_GAMES partidas (alternando color) y retorna (win_pct, draws)."""
    wins = 0
    draws = 0
    for i in range(TEST_GAMES):
        santi_color = -1 if i % 2 == 0 else 1
        winner = run_game_test(santi, juan, santi_color)
        if winner == santi_color:
            wins += 1
        elif winner == 0:
            draws += 1
    win_pct = wins / TEST_GAMES
    return win_pct, draws


# ── Persistencia ──────────────────────────────────────────────────────────────

def save_qtable(santi: FVMCPolicy) -> None:
    with open(SANTI_CACHE, "wb") as f:
        pickle.dump(santi._qt, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"checkpoints": []}


def save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


# ── Gráfica ───────────────────────────────────────────────────────────────────

LEVEL_COLORS = {200: "#4e79a7", 500: "#f28e2b", 1000: "#e15759", 2000: "#76b7b2"}

def plot_progress(progress: dict) -> None:
    checkpoints = progress["checkpoints"]
    if not checkpoints:
        print("No hay datos para graficar.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    episodes = [c["total_episodes"] for c in checkpoints]
    win_pcts = [c["win_pct"] * 100 for c in checkpoints]

    # Zonas sombreadas por nivel de N
    current_n = None
    zone_start = None
    for i, c in enumerate(checkpoints):
        n = c["juan_n"]
        if n != current_n:
            if current_n is not None:
                ax.axvspan(zone_start, episodes[i], alpha=0.08,
                           color=LEVEL_COLORS.get(current_n, "gray"),
                           label=f"Juan N={current_n}")
            current_n = n
            zone_start = episodes[i]
    if current_n is not None:
        ax.axvspan(zone_start, episodes[-1], alpha=0.08,
                   color=LEVEL_COLORS.get(current_n, "gray"),
                   label=f"Juan N={current_n}")

    # Línea de win %
    ax.plot(episodes, win_pcts, color="#333333", linewidth=2, marker="o",
            markersize=4, label="Win % Santi")

    # Target
    ax.axhline(WIN_TARGET * 100, color="green", linestyle="--", linewidth=1.5,
               label=f"Target {int(WIN_TARGET*100)}%")

    ax.set_xlabel("Episodios de entrenamiento acumulados", fontsize=12)
    ax.set_ylabel("Win % de Santi (en 10 partidas)", fontsize=12)
    ax.set_title("Progreso de Santi (FVMC) vs Juan (Flat Monte Carlo)", fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=150)
    print(f"\nGrafica guardada en: {GRAPH_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Entrenamiento progresivo: Santi vs Juan")
    print("=" * 60)

    progress = load_progress()
    total_episodes = (
        progress["checkpoints"][-1]["total_episodes"]
        if progress["checkpoints"] else 0
    )

    # Instanciar Santi sin cache automático (lo manejamos nosotros)
    santi = FVMCPolicy(cache="")  # cache="" → no carga ni guarda automáticamente

    # Cargar Q-table existente o entrenar desde cero
    if os.path.exists(SANTI_CACHE):
        print(f"\nCargando Q-table de Santi desde '{SANTI_CACHE}'...")
        with open(SANTI_CACHE, "rb") as f:
            santi._qt = pickle.load(f)
        print(f"  Estados conocidos: {len(santi._qt):,}")
    else:
        print("\nNo hay Q-table guardada. Entrenando Santi desde cero...")
        print(f"  Entrenamiento base: {santi.n_vs_random:,} vs random + "
              f"{santi.n_self_play:,} self-play")
        for i in range(santi.n_vs_random):
            santi._ep_vs_random(-1 if i % 2 == 0 else 1)
        for _ in range(santi.n_self_play):
            santi._ep_self_play()
        save_qtable(santi)
        print(f"  Q-table base guardada. Estados: {len(santi._qt):,}")

    # Determinar desde qué nivel de N retomar
    last_n = (
        progress["checkpoints"][-1]["juan_n"]
        if progress["checkpoints"] else JUAN_N_LEVELS[0]
    )
    start_level_idx = JUAN_N_LEVELS.index(last_n) if last_n in JUAN_N_LEVELS else 0

    # Handler para Ctrl+C
    interrupted = [False]
    def _handler(sig, frame):
        print("\n\nInterrumpido. Guardando y generando grafica...")
        interrupted[0] = True
    signal.signal(signal.SIGINT, _handler)

    # Bucle principal por niveles de N de Juan
    for level_idx in range(start_level_idx, len(JUAN_N_LEVELS)):
        juan_n = JUAN_N_LEVELS[level_idx]
        juan = FlatMonteCarlo(N=juan_n)

        print(f"\n{'-'*60}")
        print(f"  Nivel: Juan N={juan_n}")
        print(f"  Total episodios hasta ahora: {total_episodes:,}")
        print(f"{'-'*60}")

        for batch in range(MAX_BATCHES):
            if interrupted[0]:
                break

            # Entrenamiento: BATCH_SIZE episodios alternando color de Santi
            for i in range(BATCH_SIZE):
                santi_color = -1 if i % 2 == 0 else 1
                run_game_train(santi, juan, santi_color)
            total_episodes += BATCH_SIZE

            # Evaluación
            win_pct, draws = evaluate(santi, juan)

            checkpoint = {
                "total_episodes": total_episodes,
                "juan_n": juan_n,
                "win_pct": round(win_pct, 3),
                "draws": draws,
            }
            progress["checkpoints"].append(checkpoint)
            save_progress(progress)
            save_qtable(santi)

            print(f"  Batch {batch+1:2d}/{MAX_BATCHES} | "
                  f"Episodios: {total_episodes:7,} | "
                  f"Win %: {win_pct*100:5.1f}% | "
                  f"Empates: {draws}/{TEST_GAMES} | "
                  f"Estados Q: {len(santi._qt):,}")

            if win_pct >= WIN_TARGET:
                print(f"\n  [OK] Target alcanzado ({WIN_TARGET*100:.0f}%) con Juan N={juan_n}!")
                break

        if interrupted[0]:
            break

        if level_idx < len(JUAN_N_LEVELS) - 1:
            next_n = JUAN_N_LEVELS[level_idx + 1]
            print(f"\n  Subiendo dificultad: Juan N={juan_n} -> N={next_n}")
        else:
            print("\n  Entrenamiento completado en todos los niveles!")

    # Guardar y graficar al terminar
    save_qtable(santi)
    save_progress(progress)
    plot_progress(progress)
    print("\nListo.")


if __name__ == "__main__":
    main()
