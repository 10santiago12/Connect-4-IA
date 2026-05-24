import numpy as np
import pickle, os, time
from connect4.policy import Policy
from connect4.connect_state import ConnectState


def _pack(board_pov: np.ndarray) -> bytes:
    out = 0
    for v in (board_pov.ravel() + 1).astype(np.uint8):
        out = (out << 2) | int(v)
    return out.to_bytes(11, 'big')


class QLearningAgent(Policy):

    def __init__(self, cache="qlearning_table.pkl"):
        self.cache = cache
        self._qt   = {}   # key -> [suma_0..6, conteo_0..6]  (14 floats)
        self.rng   = np.random.default_rng(42)

    def _q(self, key):
        if key not in self._qt:
            self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0)

    def _update(self, key, col, G):
        q = self._q(key)                                    # inicializa si falta
        self._qt[key][col+7] += 1.0
        self._qt[key][col]    = (q[col] + 0.25*(G - q[col])) * self._qt[key][col+7]

    def _pick(self, key, free, eps=0.0):
        if self.rng.random() < eps: return int(self.rng.choice(free))
        q = self._q(key)
        return int(max(free, key=lambda c: q[c]))

    def _episode(self, agent=None):
        state, trajs = ConnectState(), {-1: [], 1: []}
        while not state.is_final():
            p    = state.player
            key  = _pack(state.board * p)
            col  = self._pick(key, state.get_free_cols(), eps=0.15) \
                   if (agent is None or p == agent) \
                   else int(self.rng.choice(state.get_free_cols()))
            trajs[p].append((key, col))
            state = state.transition(col)
        w = state.get_winner()
        for p in ([-1, 1] if agent is None else [agent]):
            G = 1.0 if w == p else (0.1 if w == 0 else -1.0)
            for key, col in reversed(trajs[p]):
                self._update(key, col, G)
                G *= 0.95

    def mount(self, *args, **kw):
        if self.cache and os.path.exists(self.cache):
            with open(self.cache, "rb") as f: self._qt = pickle.load(f)
            return
        ok = lambda: True
        if args and isinstance(args[0], (int, float)):
            t = time.monotonic() + args[0]*0.85
            ok = lambda: time.monotonic() < t
        for i in range(80_000):
            if not ok(): break
            self._episode(agent=-1 if i%2==0 else 1)
        for _ in range(30_000):
            if not ok(): break
            self._episode()
        if self.cache:
            with open(self.cache, "wb") as f: pickle.dump(self._qt, f, protocol=5)

    def act(self, s: np.ndarray) -> int:
        board = np.asarray(s)
        my    = -1 if np.sum(board==-1) == np.sum(board==1) else 1
        opp   = -my
        cs    = ConnectState(board, my)
        free  = cs.get_free_cols()
        if not free: return 0
        ob    = ConnectState(board, opp)

        # 1. Ganar ya
        for c in free:
            if cs.is_applicable(c) and cs.transition(c).get_winner() == my: return c
        # 2. Bloquear
        for c in free:
            if ob.is_applicable(c) and ob.transition(c).get_winner() == opp: return c
        # 3. No regalar victoria
        def dangerous(c):
            if not cs.is_applicable(c): return True
            a = cs.transition(c)
            oa = ConnectState(a.board, opp)
            return any(oa.is_applicable(x) and oa.transition(x).get_winner()==opp
                       for x in a.get_free_cols())
        cands = [c for c in free if not dangerous(c)] or free
        # 4. Q-table
        return self._pick(_pack(board*my), cands)