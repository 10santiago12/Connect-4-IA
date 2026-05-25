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

    GAMMA = 0.95
    ALPHA = 0.25

    def __init__(self, cache="qlearning_table.pkl"):
        self.cache = cache
        self._qt   = {}
        self.rng   = np.random.default_rng(42)

    # ── Tabla Q ───────────────────────────────────────────────────────────────

    def _q(self, key):
        if key not in self._qt:
            self._qt[key] = np.array([0.5]*7 + [1.0]*7, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0)

    def _update(self, key, col, target):
        q = self._q(key)
        self._qt[key][col+7] += 1.0
        self._qt[key][col]    = (q[col] + self.ALPHA*(target - q[col])) * self._qt[key][col+7]

    def _pick(self, key, free, eps=0.0):
        if self.rng.random() < eps: return int(self.rng.choice(free))
        q = self._q(key)
        return int(max(free, key=lambda c: q[c]))

    # ── Episodio con Q-Learning (bootstrapping) ───────────────────────────────

    def _episode(self, agent=None):
        """
        En cada paso calcula el target con bootstrapping:
            target = r + gamma * max_a' q(s', a')
        A diferencia de Monte Carlo, no espera al final del episodio:
        cada actualizacion ya considera las jugadas futuras estimadas.
        El max del siguiente estado se niega porque es perspectiva del oponente.
        """
        state = ConnectState()
        while not state.is_final():
            p, free = state.player, state.get_free_cols()
            key  = _pack(state.board * p)
            col  = self._pick(key, free, eps=0.15) if (agent is None or p == agent) \
                   else int(self.rng.choice(free))
            s2   = state.transition(col)
            if s2.is_final():
                w      = s2.get_winner()
                target = 1.0 if w == p else (0.1 if w == 0 else -1.0)
            else:
                # Mejor Q del siguiente estado desde perspectiva del oponente,
                # negado porque lo que es bueno para el oponente es malo para mi
                opp_best = max(self._q(_pack(s2.board * (-p)))[c]
                               for c in s2.get_free_cols())
                target   = self.GAMMA * (-opp_best)
            if agent is None or p == agent:
                self._update(key, col, target)
            state = s2

    # ── Policy API ────────────────────────────────────────────────────────────

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

        # Regla 1: ganar ya
        for c in free:
            if cs.is_applicable(c) and cs.transition(c).get_winner() == my: return c
        # Regla 2: bloquear
        for c in free:
            if ob.is_applicable(c) and ob.transition(c).get_winner() == opp: return c

        # Regla 3: puntuar cada columna mirando 3 jugadas adelante
        def score(c):
            if not cs.is_applicable(c): return -999
            s1 = cs.transition(c)
            # Contar cuantas amenazas tiene el oponente tras mi jugada
            opp1    = ConnectState(s1.board, opp)
            threats = sum(1 for c2 in s1.get_free_cols()
                          if opp1.is_applicable(c2) and opp1.transition(c2).get_winner() == opp)
            # Penalizar amenaza doble (oponente gana sin importar lo que haga)
            if threats >= 2: return -100
            # Bonificar si yo puedo ganar en mi siguiente turno (3 jugadas en total)
            wins = sum(1 for c2 in s1.get_free_cols()
                       if opp1.is_applicable(c2)
                       for s2 in [opp1.transition(c2)]
                       for me2 in [ConnectState(s2.board, my)]
                       for c3 in s2.get_free_cols()
                       if me2.is_applicable(c3) and me2.transition(c3).get_winner() == my)
            return wins - 10 * threats

        cands = sorted(free, key=score, reverse=True)
        # Regla 4: Q-table sobre candidatos (descartando los muy peligrosos)
        safe  = [c for c in cands if score(c) > -50] or cands
        return self._pick(_pack(board * my), safe)