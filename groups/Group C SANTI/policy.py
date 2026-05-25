import numpy as np
import pickle, os, time
from connect4.policy import Policy
from connect4.connect_state import ConnectState


def _pack(board_pov: np.ndarray) -> bytes:
    out = 0
    for v in (board_pov.ravel() + 1).astype(np.uint8):
        out = (out << 2) | int(v)
    return out.to_bytes(11, 'big')


class FVMCPolicy(Policy):
    """
    FVMC + retorno con descuento hacia atrás (backward pass).
    
    Mejora clave sobre versión anterior:
    - Al actualizar la trayectoria, recorre los movimientos en reversa
      aplicando G *= gamma en cada paso. Esto propaga el resultado final
      hacia los movimientos del medio de la partida — el agente aprende
      que ciertos estados intermedios son valiosos, no solo el último movimiento.
    - gamma=0.95 (igual que el Q-learning del compañero) para comparación justa.
    - Más episodios de self-play que vs random para aprender posiciones complejas.
    """

    def __init__(self, n_vs_random=80_000, n_self_play=40_000,
                 gamma=0.95, cache="fvmc_qtable.pkl"):
        self.n_vs_random = n_vs_random
        self.n_self_play = n_self_play
        self.gamma = gamma
        self.cache = cache
        self._qt: dict[bytes, np.ndarray] = {}

    # ── Q-table ───────────────────────────────────────────────────────────────

    def _q(self, key: bytes) -> np.ndarray:
        if key not in self._qt:
            self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0)

    def _update(self, traj: list, G_final: float) -> None:
        """
        Backward pass con descuento: recorre la trayectoria en reversa.
        Cada estado recibe G = G_siguiente * gamma, en vez de todos recibir
        el mismo G_final. Esto premia los movimientos que llevaron a ganar
        y penaliza los que llevaron a perder, con más peso en los recientes.
        First-Visit: solo actualiza la primera vez que aparece cada (estado, acción).
        """
        seen = set()
        G = G_final
        for key, col in reversed(traj):        # <-- reversa + descuento
            if (key, col) not in seen:
                seen.add((key, col))
                if key not in self._qt:
                    self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
                self._qt[key][col] += G
                self._qt[key][col + 7] += 1.0
            G *= self.gamma                    # <-- descuento hacia atrás

    # ── exploración ───────────────────────────────────────────────────────────

    def _explore(self, board_pov: np.ndarray, free: list) -> int:
        q = self._q(_pack(board_pov))
        w = np.maximum(q[free], 1e-6); w /= w.sum()
        return int(np.random.choice(free, p=w))

    # ── episodios ─────────────────────────────────────────────────────────────

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

    def mount(self, *args, **_kw) -> None:
        if self.cache and os.path.exists(self.cache):
            with open(self.cache, "rb") as f: self._qt = pickle.load(f)
            return
        deadline = time.monotonic() + args[0] if args and isinstance(args[0], (int, float)) else None
        done = lambda: deadline and time.monotonic() >= deadline
        for i in range(self.n_vs_random):
            if done(): return
            self._ep_vs_random(-1 if i % 2 == 0 else 1)
        for _ in range(self.n_self_play):
            if done(): return
            self._ep_self_play()
        if self.cache:
            with open(self.cache, "wb") as f:
                pickle.dump(self._qt, f, protocol=pickle.HIGHEST_PROTOCOL)

    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s)
        my = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
        bpov = board * my
        cs = ConnectState(board, my)
        if cs.is_final(): return 0
        free = cs.get_free_cols()
        if not free: return 0
        # Ganar inmediato / bloquear
        for col in free:
            if ConnectState(board, my).transition(col).get_winner() == my: return col
        for col in free:
            ob = ConnectState(board, -my)
            if ob.is_applicable(col) and ob.transition(col).get_winner() == -my: return col
        # Q-table greedy
        q = self._q(_pack(bpov))
        return int(max(free, key=lambda c: q[c]))