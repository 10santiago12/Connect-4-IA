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
    Agente Connect-4 — First-Visit Monte Carlo + GPI iterativo (clases 11-12).

    Mejoras sobre versión anterior:
    - Se eliminó el filtro gives_opp_win: el Q-table se ve forzado a aprender
      posiciones peligrosas por sí solo, mejorando Q-values más allá de 1 jugada.
    - GPI iterativo: n_rounds rondas de (vs_random → self_play). Cada ronda el
      oponente en self_play es más fuerte, los Q-values reflejan amenazas más
      profundas iteración a iteración.
    - Rondas 2+ usan menos episodios vs_random (refinamiento, no re-aprendizaje).
    """

    def __init__(self, n_vs_random=60_000, n_self_play=40_000,
                 n_rounds=3, cache="fvmc_qtable.pkl"):
        self.n_vs_random = n_vs_random
        self.n_self_play = n_self_play
        self.n_rounds = n_rounds
        self.cache = cache
        self._qt: dict[bytes, np.ndarray] = {}

    # ── Q-table ops ───────────────────────────────────────────────────────────

    def _q(self, key: bytes) -> np.ndarray:
        if key not in self._qt:
            self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0)

    def _update(self, traj: list, G: float) -> None:
        seen = set()
        for key, col in traj:
            if (key, col) in seen: continue
            seen.add((key, col))
            if key not in self._qt:
                self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
            self._qt[key][col] += G
            self._qt[key][col + 7] += 1.0

    # ── acciones ──────────────────────────────────────────────────────────────

    def _explore(self, board_pov: np.ndarray, free: list) -> int:
        """Win-probability sampling durante entrenamiento: π(a|s) ∝ Q(s,a)."""
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

        for rnd in range(self.n_rounds):
            # Ronda 1: aprendizaje base vs random. Rondas 2+: solo refinamiento.
            eps_r = self.n_vs_random if rnd == 0 else self.n_vs_random // 4
            for i in range(eps_r):
                if done(): return
                self._ep_vs_random(-1 if i % 2 == 0 else 1)
            for _ in range(self.n_self_play):
                if done(): break
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

        # Ganar inmediato / bloquear victoria inmediata del oponente
        for col in free:
            if ConnectState(board, my).transition(col).get_winner() == my: return col
        for col in free:
            opp_s = ConnectState(board, -my)
            if opp_s.is_applicable(col) and opp_s.transition(col).get_winner() == -my: return col

        # Q-table greedy — decide todo lo que no es victoria/bloqueo inmediato
        q = self._q(_pack(bpov))
        return int(max(free, key=lambda c: q[c]))