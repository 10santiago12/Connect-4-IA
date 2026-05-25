"""
generate_data.py
Corre todos los experimentos y guarda results.json.
Ejecutar UNA vez antes de abrir el notebook.
"""

import json, time, numpy as np
import sys
sys.path.insert(0, "groups/Group C SANTI")
from connect4.connect_state import ConnectState
from policy import FVMCPolicy

RNG = np.random.default_rng(0)

# ── helpers ────────────────────────────────────────────────────────────────

def random_opp(state, _):
    return int(RNG.choice(state.get_free_cols()))

def evaluate(agent, opp_fn, n_games=300):
    wins = draws = losses = 0
    for i in range(n_games):
        color = 1 if i % 2 == 0 else -1
        state = ConnectState()
        while not state.is_final():
            col = agent.act(state.board) if state.player == color else opp_fn(state, state.player)
            state = state.transition(col)
        w = state.get_winner()
        if w == color:  wins += 1
        elif w == 0:    draws += 1
        else:           losses += 1
    return {"win_pct": wins/n_games, "draw_pct": draws/n_games, "loss_pct": losses/n_games}

def evaluate_by_color(agent, opp_fn, n_games=200):
    result = {}
    for color, label in [(1, "yellow"), (-1, "red")]:
        wins = 0
        for _ in range(n_games):
            state = ConnectState()
            while not state.is_final():
                col = agent.act(state.board) if state.player == color else opp_fn(state, state.player)
                state = state.transition(col)
            if state.get_winner() == color: wins += 1
        result[label] = wins / n_games
    return result

def self_play_eval(agent, n_games=400):
    red_wins = yellow_wins = draws = 0
    for _ in range(n_games):
        state = ConnectState()
        while not state.is_final():
            state = state.transition(agent.act(state.board))
        w = state.get_winner()
        if w == 1:    yellow_wins += 1
        elif w == -1: red_wins += 1
        else:         draws += 1
    total = n_games
    return {"red_wins": red_wins, "yellow_wins": yellow_wins, "draws": draws,
            "red_pct": red_wins/total, "yellow_pct": yellow_wins/total}

def train_fresh(n_vs_random, n_self_play, heuristic_w=0.4, seed=42):
    """Entrena un agente desde cero sin caché."""
    a = FVMCPolicy(n_vs_random=n_vs_random, n_self_play=n_self_play,
                   heuristic_w=heuristic_w, cache=None, seed=seed)
    a.mount()
    return a

# ── Experimento 1: curva de aprendizaje ────────────────────────────────────
print("Exp 1: curva de aprendizaje...")
checkpoints = [0, 5_000, 10_000, 20_000, 40_000, 60_000, 80_000, 100_000, 120_000]
exp1 = []
agent = FVMCPolicy(n_vs_random=0, n_self_play=0, cache=None, seed=42)
agent.mount()

prev = 0
for ck in checkpoints:
    delta = ck - prev
    for i in range(delta):
        agent._episode(-1 if i % 2 == 0 else 1,
                       opp=lambda s, _: int(RNG.choice(s.get_free_cols())))
    prev = ck
    r = evaluate(agent, random_opp, n_games=300)
    exp1.append({"episodes": ck, **r})
    print(f"  {ck:>7,} ep → win={r['win_pct']:.2%}  loss={r['loss_pct']:.2%}")

# ── Experimento 2: heuristic_w sweep ───────────────────────────────────────
print("Exp 2: impacto de heuristic_w...")
hw_values = [0.0, 0.2, 0.4, 0.6, 0.8]
exp2 = {}
for hw in hw_values:
    a = train_fresh(40_000, 80_000, heuristic_w=hw)
    r = evaluate(a, random_opp, n_games=300)
    exp2[f"hw_{hw}"] = {"heuristic_w": hw, **r}
    print(f"  hw={hw} → win={r['win_pct']:.2%}  loss={r['loss_pct']:.2%}")

# ── Experimento 3: self-play ────────────────────────────────────────────────
print("Exp 3: self-play...")
agent_final = train_fresh(40_000, 80_000, heuristic_w=0.4)
exp3 = self_play_eval(agent_final, n_games=400)
print(f"  Rojo={exp3['red_pct']:.2%}  Amarillo={exp3['yellow_pct']:.2%}  Empate={exp3['draws']/400:.2%}")

# ── Experimento 4: desempeño por color ─────────────────────────────────────
print("Exp 4: desempeño por color...")
exp4 = evaluate_by_color(agent_final, random_opp, n_games=200)
print(f"  Amarillo={exp4['yellow']:.2%}  Rojo={exp4['red']:.2%}")

# ── Guardar ────────────────────────────────────────────────────────────────
results = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M"),
    "mode": "full",
    "exp1_learning_curve": exp1,
    "exp2_heuristic_w":    list(exp2.values()),
    "exp3_self_play":      exp3,
    "exp4_by_color":       exp4,
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Listo → results.json")