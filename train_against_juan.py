"""
Entrenamiento progresivo: Santi (FVMCPolicy) vs Juan (FlatMonteCarlo).

Juan N=200 tarda ~42s por partida, así que los batches son pequeños (20 ep)
pero suficientes para medir progreso. El script es interrumpible con Ctrl+C
y retoma desde donde quedó gracias al cache y al progress.json.

Uso:
    python train_against_juan.py

Archivos generados:
    fvmc_qtable.pkl        — Q-table de Santi (se actualiza cada batch)
    training_progress.json — historial de checkpoints
    training_progress.png  — gráfica de progreso
"""

import sys, os, json, pickle, signal, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ajusta estas rutas a tu estructura de carpetas ────────────────────────────
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from connect4.connect_state import ConnectState

# Importar políticas — ajusta los nombres de carpeta si difieren
import importlib
FVMCPolicy    = importlib.import_module("groups.Group C SANTI.policy").FVMCPolicy
FlatMonteCarlo = importlib.import_module("groups.Group A JUAN.policy").FlatMonteCarlo

# ── configuración ─────────────────────────────────────────────────────────────
# Juan N=200 → ~42s/partida. Con BATCH=20 cada checkpoint tarda ~14 min.
# Sube BATCH_SIZE solo si tu máquina es más rápida que eso.

JUAN_N_LEVELS = [200, 500]      # N=1000/2000 son inviables en tiempo
BATCH_SIZE    = 20              # episodios por checkpoint
TEST_GAMES    = 20              # partidas de prueba (alternando color)
WIN_TARGET    = 0.60            # target más realista que 70% contra Flat MC
MAX_BATCHES   = 30              # máximo de batches por nivel

SANTI_CACHE   = "fvmc_qtable.pkl"
PROGRESS_FILE = "training_progress.json"
GRAPH_FILE    = "training_progress.png"

# ── helpers ───────────────────────────────────────────────────────────────────

def _pack(board_pov: np.ndarray) -> bytes:
    out = 0
    for v in (board_pov.ravel() + 1).astype(np.uint8):
        out = (out << 2) | int(v)
    return out.to_bytes(11, 'big')


def run_episode_train(santi, juan, santi_color: int) -> None:
    """Un episodio de entrenamiento: Santi explora, Juan juega greedy."""
    state, traj, first = ConnectState(), [], True
    while not state.is_final():
        free = state.get_free_cols()
        if state.player == santi_color:
            bpov = state.board * santi_color
            col = int(np.random.choice(free)) if first else santi._explore(bpov, free)
            first = False
            traj.append((_pack(bpov), col))
        else:
            col = juan.act(state.board)
        state = state.transition(col)
    w = state.get_winner()
    santi._update(traj, 1.0 if w == santi_color else (0.0 if w == 0 else -1.0))


def evaluate(santi, juan, n=TEST_GAMES) -> dict:
    """Mide win/loss/draw de Santi contra Juan en n partidas."""
    wins, losses, draws = 0, 0, 0
    for i in range(n):
        santi_color = -1 if i % 2 == 0 else 1
        state = ConnectState()
        while not state.is_final():
            col = santi.act(state.board) if state.player == santi_color else juan.act(state.board)
            state = state.transition(col)
        w = state.get_winner()
        if w == santi_color:   wins += 1
        elif w == 0:           draws += 1
        else:                  losses += 1
    return {"wins": wins, "losses": losses, "draws": draws,
            "win_pct": wins / n, "loss_pct": losses / n}


def save_qtable(santi) -> None:
    with open(SANTI_CACHE, "wb") as f:
        pickle.dump(santi._qt, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f: return json.load(f)
    return {"checkpoints": []}


def save_progress(p: dict) -> None:
    with open(PROGRESS_FILE, "w") as f: json.dump(p, f, indent=2)


def plot_progress(progress: dict) -> None:
    pts = progress["checkpoints"]
    if not pts: return

    eps      = [p["total_episodes"] for p in pts]
    win_pcts = [p["win_pct"] * 100  for p in pts]
    los_pcts = [p["loss_pct"] * 100 for p in pts]

    fig, ax = plt.subplots(figsize=(11, 5))

    # Zonas sombreadas por nivel de N de Juan
    colors = {200: "#4e79a7", 500: "#f28e2b", 1000: "#e15759"}
    prev_n, zone_start = None, eps[0]
    for i, p in enumerate(pts):
        n = p["juan_n"]
        if n != prev_n:
            if prev_n is not None:
                ax.axvspan(zone_start, eps[i], alpha=0.07,
                           color=colors.get(prev_n, "gray"), label=f"Juan N={prev_n}")
            prev_n, zone_start = n, eps[i]
    ax.axvspan(zone_start, eps[-1], alpha=0.07,
               color=colors.get(prev_n, "gray"), label=f"Juan N={prev_n}")

    ax.plot(eps, win_pcts, color="#2ecc71", lw=2, marker="o", ms=4, label="Win %")
    ax.plot(eps, los_pcts, color="#e74c3c", lw=2, marker="o", ms=4, label="Loss %")
    ax.axhline(WIN_TARGET * 100, color="green", ls="--", lw=1.5,
               label=f"Target {int(WIN_TARGET*100)}%")

    ax.set_xlabel("Episodios acumulados de entrenamiento vs Juan")
    ax.set_ylabel("% partidas (sobre 20 pruebas)")
    ax.set_title("Progreso de Santi (FVMC) entrenando contra Juan (Flat MC)")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=150)
    print(f"Gráfica guardada en: {GRAPH_FILE}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Entrenamiento Santi (FVMC) vs Juan (Flat MC)")
    print("=" * 60)

    progress = load_progress()
    total_ep = progress["checkpoints"][-1]["total_episodes"] if progress["checkpoints"] else 0

    # Instanciar Santi sin cache automático (lo manejamos aquí)
    santi = FVMCPolicy(cache="")

    # Cargar Q-table o entrenar base
    if os.path.exists(SANTI_CACHE):
        with open(SANTI_CACHE, "rb") as f: santi._qt = pickle.load(f)
        print(f"Q-table cargada — {len(santi._qt):,} estados")
    else:
        print("Entrenando Q-table base (vs random + self-play)...")
        for i in range(santi.n_vs_random):
            santi._ep_vs_random(-1 if i % 2 == 0 else 1)
        for _ in range(santi.n_self_play):
            santi._ep_self_play()
        save_qtable(santi)
        print(f"Base lista — {len(santi._qt):,} estados")

    # Retomar desde el último nivel si hay progreso guardado
    last_n = progress["checkpoints"][-1]["juan_n"] if progress["checkpoints"] else JUAN_N_LEVELS[0]
    start_idx = JUAN_N_LEVELS.index(last_n) if last_n in JUAN_N_LEVELS else 0

    # Ctrl+C guarda y grafica antes de salir
    interrupted = [False]
    def _handler(sig, frame):
        print("\nInterrumpido — guardando...")
        interrupted[0] = True
    signal.signal(signal.SIGINT, _handler)

    for level_idx in range(start_idx, len(JUAN_N_LEVELS)):
        juan_n = JUAN_N_LEVELS[level_idx]
        juan   = FlatMonteCarlo(N=juan_n)
        juan.mount()

        print(f"\n{'─'*60}")
        print(f"  Nivel: Juan N={juan_n}  |  ep acumulados: {total_ep:,}")
        print(f"  Tiempo estimado por batch ({BATCH_SIZE} ep): "
              f"~{BATCH_SIZE * juan_n / 200 * 14:.0f} min")
        print(f"{'─'*60}")

        for batch in range(MAX_BATCHES):
            if interrupted[0]: break
            t0 = time.time()

            for i in range(BATCH_SIZE):
                run_episode_train(santi, juan, -1 if i % 2 == 0 else 1)
            total_ep += BATCH_SIZE

            result = evaluate(santi, juan)
            elapsed = time.time() - t0

            ck = {"total_episodes": total_ep, "juan_n": juan_n,
                  "win_pct": round(result["win_pct"], 3),
                  "loss_pct": round(result["loss_pct"], 3),
                  "draws": result["draws"]}
            progress["checkpoints"].append(ck)
            save_progress(progress)
            save_qtable(santi)

            print(f"  Batch {batch+1:2d}/{MAX_BATCHES} | "
                  f"ep: {total_ep:,} | "
                  f"W: {result['win_pct']*100:.0f}% "
                  f"L: {result['loss_pct']*100:.0f}% "
                  f"D: {result['draws']}/{TEST_GAMES} | "
                  f"Q-states: {len(santi._qt):,} | "
                  f"{elapsed:.0f}s")

            if result["win_pct"] >= WIN_TARGET:
                print(f"\n  ✓ Target alcanzado con Juan N={juan_n}!")
                break

        if interrupted[0]: break

        if level_idx < len(JUAN_N_LEVELS) - 1:
            print(f"\n  Subiendo: N={juan_n} → N={JUAN_N_LEVELS[level_idx+1]}")
        else:
            print("\n  Entrenamiento completado.")

    save_qtable(santi)
    save_progress(progress)
    plot_progress(progress)


if __name__ == "__main__":
    main()