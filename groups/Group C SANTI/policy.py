import numpy as np
import pickle, os, time
from connect4.policy import Policy
from connect4.connect_state import ConnectState

def _pack(b: np.ndarray) -> bytes:
    out = 0
    for v in (b.ravel() + 1).astype(np.uint8): out = (out << 2) | int(v)
    return out.to_bytes(11, 'big')

CENTER = np.array([0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.0], dtype=np.float32)

def _heuristic(board: np.ndarray, color: int) -> float:
    """Valor parcial del tablero: cuenta amenazas de 3 en raya propias vs oponente."""
    b, rows, cols, score = board * color, 6, 7, 0.0
    def threats(sign):
        t = 0
        for r in range(rows):
            for c in range(cols - 3):
                w = b[r, c:c+4]
                if np.sum(w == sign) == 3 and np.sum(w == 0) == 1: t += 1
        for r in range(rows - 3):
            for c in range(cols):
                w = b[r:r+4, c]
                if np.sum(w == sign) == 3 and np.sum(w == 0) == 1: t += 1
        for r in range(rows - 3):
            for c in range(cols - 3):
                w = [b[r+i, c+i] for i in range(4)]
                if w.count(sign) == 3 and w.count(0) == 1: t += 1
            for c in range(3, cols):
                w = [b[r+i, c-i] for i in range(4)]
                if w.count(sign) == 3 and w.count(0) == 1: t += 1
        return t
    score = (threats(1) - threats(-1)) * 0.05
    return float(np.clip(score, -0.5, 0.5))

class FVMCPolicy(Policy):
    def __init__(self, n_vs_random=40_000, n_self_play=80_000,
                 ucb_c=0.6, heuristic_w=0.4, cache="fvmc_qtable.pkl", seed=None):
        self.n_vs_random = n_vs_random
        self.n_self_play = n_self_play
        self.ucb_c = ucb_c
        self.heuristic_w = heuristic_w  # peso de la heurística vs recompensa final
        self.cache = cache
        self._qt: dict[bytes, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    def _canon(self, b: np.ndarray) -> tuple[bytes, bool]:
        k, m = _pack(b), _pack(b[:, ::-1])
        return (m, True) if m < k else (k, False)

    def _mc(self, col: int) -> int:
        return ConnectState.COLS - 1 - col

    def _q(self, key: bytes) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._qt: self._qt[key] = np.zeros(14, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0), d[7:]

    def _update(self, traj: list[tuple[bytes, int]], g: float) -> None:
        seen = set()
        for key, col in reversed(traj):
            if (key, col) not in seen:
                seen.add((key, col))
                if key not in self._qt: self._qt[key] = np.zeros(14, dtype=np.float32)
                self._qt[key][col] += g
                self._qt[key][col + 7] += 1.0

    def _pick(self, key: bytes, free_c: list[int], exploit: bool = False) -> int:
        q, n = self._q(key)
        if exploit:
            return int(free_c[int(np.argmax(q[free_c] + CENTER[free_c]))])
        unvisited = [c for c in free_c if n[c] == 0]
        if unvisited: return int(self._rng.choice(unvisited))
        bonus = self.ucb_c * np.sqrt(np.log(np.sum(n[free_c])) / np.maximum(n[free_c], 1))
        return int(free_c[int(np.argmax(q[free_c] + bonus + CENTER[free_c]))])

    def _act_c(self, b: np.ndarray, free: list[int], exploit: bool = False) -> tuple[int, bytes, int]:
        key, mir = self._canon(b)
        fc = [self._mc(c) if mir else c for c in free]
        bc = self._pick(key, fc, exploit)
        return (self._mc(bc) if mir else bc), key, bc

    def _episode(self, color: int, opp=None) -> None:
        """color: jugador que aprende. opp=None → self-play."""
        state, trajs, first = ConnectState(), {-1: [], 1: []}, True
        while not state.is_final():
            p, free = state.player, state.get_free_cols()
            bpov = state.board * p
            if opp and p != color:
                col = opp(state, p)
            else:
                if first and p == color:
                    col = int(self._rng.choice(free))  # Exploring Starts
                    key, mir = self._canon(bpov)
                    col_c = self._mc(col) if mir else col
                    first = False
                else:
                    col, key, col_c = self._act_c(bpov, free)
                trajs[p].append((key, col_c))
            state = state.transition(col)
        w = state.get_winner()
        players = [color] if opp else [-1, 1]
        for p in players:
            r_final = (1.0 if w == p else 0.0 if w == 0 else -1.0)
            r_heur  = _heuristic(state.board, p)
            g = (1 - self.heuristic_w) * r_final + self.heuristic_w * r_heur
            self._update(trajs[p], g)

    def mount(self, *args, **_kw) -> None:
        if self._seed is None: self._rng = np.random.default_rng()
        if self.cache and os.path.exists(self.cache):
            with open(self.cache, "rb") as f: self._qt = pickle.load(f)
            return
        deadline = time.monotonic() + args[0] if args and isinstance(args[0], (int, float)) else None
        done = lambda: deadline and time.monotonic() >= deadline
        for i in range(self.n_vs_random):
            if done(): return
            self._episode(-1 if i % 2 == 0 else 1, opp=lambda s, _: int(self._rng.choice(s.get_free_cols())))
        third = self.n_self_play // 3
        for i in range(self.n_self_play):
            if done(): return
            if i < third and self._rng.random() < 0.5:
                self._episode(-1 if i % 2 == 0 else 1, opp=lambda s, _: int(self._rng.choice(s.get_free_cols())))
            else:
                self._episode(1)  # color irrelevante en self-play (trajs[p] para ambos)
        if self.cache:
            with open(self.cache, "wb") as f: pickle.dump(self._qt, f, protocol=pickle.HIGHEST_PROTOCOL)

    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s)
        my = 1 if np.sum(board == 1) <= np.sum(board == -1) else -1
        cs = ConnectState(board, my)
        if cs.is_final() or not (free := cs.get_free_cols()): return 0
        # Ganar inmediato
        for col in free:
            if ConnectState(board, my).transition(col).get_winner() == my: return col
        # Bloquear
        for col in free:
            ob = ConnectState(board, -my)
            if ob.is_applicable(col) and ob.transition(col).get_winner() == -my: return col
        # Evitar regalar victoria al turno siguiente
        safe = [c for c in free if not any(
            ConnectState(ConnectState(board, my).transition(c).board, -my).transition(c2).get_winner() == -my
            for c2 in ConnectState(board, my).transition(c).get_free_cols()
        )]
        free = safe or free
        # Q-table greedy
        key, mir = self._canon(board * my)
        q, _ = self._q(key)
        fc = [self._mc(c) if mir else c for c in free]
        return int(self._mc(fc[int(np.argmax(q[fc] + CENTER[fc]))]) if mir else fc[int(np.argmax(q[fc] + CENTER[fc]))])