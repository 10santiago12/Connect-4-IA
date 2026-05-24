import numpy as np
import pickle, os, time
from connect4.policy import Policy
from connect4.connect_state import ConnectState


def _pack(board_pov: np.ndarray) -> bytes:
    """Codifica tablero 6×7 en 11 bytes (2 bits/celda)."""
    out = 0
    for v in (board_pov.ravel() + 1).astype(np.uint8):
        out = (out << 2) | int(v)
    return out.to_bytes(11, 'big')


class FVMCPolicy(Policy):
    """
    Agente Connect-4 — First-Visit Monte Carlo + GPI (clase 11-12).

    - Perspectiva normalizada: tablero siempre desde POV del jugador activo
    → un Q-table compartido para ambos colores (Alternating Markov Game).
    - Exploración por win-probability: π(a|s) ∝ Q(s,a).
    - Almacenamiento compacto: key 11 bytes (bit-pack) + array float32 fusionado
    sum_0..sum_6 | count_0..count_6] por estado.
    """

    def __init__(self, n_vs_random=60_000, n_self_play=20_000, cache="fvmc_qtable.pkl"):
        self.n_vs_random = n_vs_random
        self.n_self_play = n_self_play
        self.cache = cache
        self._qt: dict[bytes, np.ndarray] = {}  # key → float32(14,)

    # ── Q-table ops ───────────────────────────────────────────────────────────
    
    def _q(self, key: bytes) -> np.ndarray:
        if key not in self._qt:
            self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0) 
    #Recorre todos los movimientos de la trayectoria, actualizando la Q-table con la recompensa G obtenida al final del episodio. Para cada estado (key) y acción (col), acumula la suma de recompensas y el conteo de visitas para calcular la probabilidad de victoria.
    def _update(self, traj: list, G: float) -> None:
        seen = set()
        for key, col in traj:
            if (key, col) in seen: continue
            seen.add((key, col))
            if key not in self._qt:
                self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
            self._qt[key][col] += G
            self._qt[key][col + 7] += 1.0

    # ── acción exploratoria ───────────────────────────────────────────────────
    
    #le da mas peso a las acciones con mayor probabilidad de victoria, pero mantiene algo de aleatoriedad para explorar el espacio de estados.
    def _explore(self, board_pov: np.ndarray, free: list) -> int:
        q = self._q(_pack(board_pov))
        w = np.maximum(q[free], 1e-6); w /= w.sum()
        return int(np.random.choice(free, p=w))

    # ── episodios ─────────────────────────────────────────────────────────────

    #Vs agente random : alternar colores cada episodio para balancear experiencia.

    def _ep_vs_random(self, color: int) -> None:
        state, traj, first = ConnectState(), [], True
        while not state.is_final():
            free = state.get_free_cols()
            if state.player == color:
                bpov = state.board * color
                col = int(np.random.choice(free)) if first else self._explore(bpov, free)
                first = False
                traj.append((_pack(bpov), col))
            else:
                col = int(np.random.choice(free))
            state = state.transition(col)
        w = state.get_winner()
        self._update(traj, 1.0 if w == color else (0.0 if w == 0 else -1.0))
    
    #Vs el mismo agente: ambos jugadores siguen la política actual. 

    def _ep_self_play(self) -> None:
        state, trajs = ConnectState(), {-1: [], 1: []}
        while not state.is_final():
            p = state.player
            bpov = state.board * p
            free = state.get_free_cols()
            col = self._explore(bpov, free)
            trajs[p].append((_pack(bpov), col))
            state = state.transition(col)
        w = state.get_winner()
        for p in (-1, 1):
            self._update(trajs[p], 1.0 if w == p else (0.0 if w == 0 else -1.0))

    # ── Policy API ────────────────────────────────────────────────────────────

    #Montar la política: generar experiencia y entrenar Q-table (guardada en archivo .pkl). Si se da un argumento numérico, usarlo como timeout (segundos).

    def mount(self, *args, **_kw) -> None:
        if self.cache and os.path.exists(self.cache):
            with open(self.cache, "rb") as f: self._qt = pickle.load(f)
            return
        deadline = time.monotonic() + args[0] if args and isinstance(args[0], (int, float)) else None
        done = lambda: deadline and time.monotonic() >= deadline
        for i in range(self.n_vs_random):
            if done(): break
            self._ep_vs_random(-1 if i % 2 == 0 else 1)
        for _ in range(self.n_self_play):
            if done(): break
            self._ep_self_play()
        if self.cache:
            with open(self.cache, "wb") as f: pickle.dump(self._qt, f, protocol=pickle.HIGHEST_PROTOCOL)

    #La que toma decisiones reales durante elk juego: filtrar jugadas perdedoras, luego ganar/bloquear, luego greedy Q-table.

>>>>>>> origin/main
    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s)
        my = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
        bpov = board * my
        cs = ConnectState(board, my)
        if cs.is_final(): return 0
        free = cs.get_free_cols()
        if not free: return 0

        # Filtrar jugadas que regalan victoria al oponente
        def gives_opp_win(col):
            after = ConnectState(board, my).transition(col)
            opp = -my
            for c in after.get_free_cols():
                s = ConnectState(after.board, opp)
                if s.is_applicable(c) and s.transition(c).get_winner() == opp:
                    return True
            return False
        safe = [c for c in free if not gives_opp_win(c)]
        cands = safe if safe else free

        # 1. Si puedo ganar en esta jugada, la hago
        for col in cands:
            if ConnectState(board, my).transition(col).get_winner() == my: return col
        # 2. Si el oponente puede ganar en la siguiente jugada, bloqueo esa jugada
        for col in cands:
            if ConnectState(board, -my).is_applicable(col) and \
               ConnectState(board, -my).transition(col).get_winner() == -my: return col
        # 3. Consultar Q-table para elegir la jugada con mayor probabilidad de victoria
        q = self._q(_pack(bpov))
        return int(max(cands, key=lambda c: q[c]))