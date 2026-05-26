"""
generate_data.py
Corre todos los experimentos y guarda results.json.
Ejecutar UNA vez antes de abrir el notebook.
"""

import json, time, numpy as np
import importlib.util
import sys
sys.path.insert(0, "groups/Group C SANTI")
from connect4.connect_state import ConnectState
from policy import FVMCPolicy

def _load_policy_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location(class_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)

FlatMonteCarlo = _load_policy_class("groups/Group A JUAN/policy.py", "FlatMonteCarlo")
QLearningAgent = _load_policy_class("groups/Group B GUTI/policy.py", "QLearningAgent")

RNG = np.random.default_rng(0)
main_ep = 120_000
n_bench = 300


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

def evaluate_head_to_head(agent_a, agent_b, name_a, name_b, n_games=200):
    """Enfrenta dos agentes ya montados, alternando colores."""
    results = {name_a: 0, name_b: 0, "Draw": 0}
    detail = []
    for i in range(n_games):
        if i % 2 == 0:
            red, red_name = agent_a, name_a
            yel, yel_name = agent_b, name_b
        else:
            red, red_name = agent_b, name_b
            yel, yel_name = agent_a, name_a

        state = ConnectState()
        while not state.is_final():
            col = red.act(state.board) if state.player == -1 else yel.act(state.board)
            state = state.transition(col)

        w = state.get_winner()
        winner = red_name if w == -1 else (yel_name if w == 1 else "Draw")
        results[winner] += 1
        detail.append(winner)

    return results, detail

def _cumulative_wr(detail, winner_name):
    running = 0
    out = []
    for i, w in enumerate(detail, 1):
        if w == winner_name:
            running += 1
        out.append(round(running / i, 4))
    return out

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
hw_values = [0.2, 0.4, 0.6]
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

# ── Experimento 5: cobertura de Q-table por turno ──────────────────────────
def exp5_qtable_coverage(FVMCPolicy, n_ep, n_eval=500):
    print("\n[5] Cobertura de Q-table por turno...")
    agent = FVMCPolicy(n_vs_random=int(n_ep*0.67),
                       n_self_play=n_ep-int(n_ep*0.67),
                       heuristic_w=0.4, cache=None, seed=42)
    agent.mount()
    
    coverage_by_turn = {}  # turno -> [visto, total]
    rng = np.random.default_rng(0)
    
    for i in range(n_eval):
        color = 1 if i % 2 == 0 else -1
        state = ConnectState()
        turn = 0
        while not state.is_final():
            turn += 1
            if state.player == color:
                bpov = state.board * color
                key, _ = agent._canon(bpov)
                seen = key in agent._qt
                if turn not in coverage_by_turn:
                    coverage_by_turn[turn] = [0, 0]
                coverage_by_turn[turn][1] += 1
                if seen:
                    coverage_by_turn[turn][0] += 1
                col = agent.act(state.board)
            else:
                col = int(rng.choice(state.get_free_cols()))
            state = state.transition(col)
    
    result = [
        {"turn": t, "coverage_pct": round(v[0]/v[1], 4)}
        for t, v in sorted(coverage_by_turn.items())
    ]
    for r in result:
        print(f"  Turno {r['turn']:>2}: {r['coverage_pct']*100:.1f}%")
    return result

# ── Experimento 6: calidad de Q-values vs trials totales ───────────────────
def exp6_trials_sweep(FVMCPolicy, n_bench=300):
    print("\n[6] Impacto de trials totales...")
    trials_list = [15_000, 60_000, 120_000]
    results = []
    for n_ep in trials_list:
        agent = FVMCPolicy(n_vs_random=int(n_ep*0.67),
                           n_self_play=n_ep-int(n_ep*0.67),
                           heuristic_w=0.4, cache=None, seed=42)
        agent.mount()
        
        # win rate por color
        wins = {1: 0, -1: 0}
        for i in range(n_bench):
            color = 1 if i % 2 == 0 else -1
            state = ConnectState()
            rng = np.random.default_rng(i)
            while not state.is_final():
                col = agent.act(state.board) if state.player == color \
                      else int(rng.choice(state.get_free_cols()))
                state = state.transition(col)
            if state.get_winner() == color:
                wins[color] += 1
        
        # tamaño de la tabla
        qt_size = len(agent._qt)
        
        wr_y = wins[1] / (n_bench // 2)
        wr_r = wins[-1] / (n_bench // 2)
        results.append({
            "total_trials": n_ep,
            "wr_yellow": round(wr_y, 4),
            "wr_red":    round(wr_r, 4),
            "wr_avg":    round((wr_y + wr_r) / 2, 4),
            "qt_size":   qt_size
        })
        print(f"  {n_ep:>7,} trials → WR amarillo={wr_y*100:.1f}%  rojo={wr_r*100:.1f}%  tabla={qt_size:,} estados")
    return results
# ── Guardar ────────────────────────────────────────────────────────────────
results = {
    "generated_at": time.strftime("%Y-%m-%d %H:%M"),
    "mode": "full",
    "exp1_learning_curve": exp1,
    "exp2_heuristic_w":    list(exp2.values()),
    "exp3_self_play":      exp3,
    "exp4_by_color":       exp4,
    "exp5_qtable_coverage": exp5_qtable_coverage(FVMCPolicy, main_ep),
    "exp6_trials_sweep":    exp6_trials_sweep(FVMCPolicy),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Listo → results.json")