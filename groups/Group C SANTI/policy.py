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

    def __init__(self, n_vs_random=18_000, n_self_play=12_000, n_vs_heuristic=10_000,
                 gamma=0.95, cache="fvmc_qtable.pkl",
                 eps_start=0.35, eps_end=0.05, seed=0,
                 shape_threat=0.03, shape_center=0.02, shape_opp_win=0.04):
        self.n_vs_random = n_vs_random
        self.n_self_play = n_self_play
        self.n_vs_heuristic = n_vs_heuristic
        self.gamma = gamma
        self.cache = cache
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.shape_threat = shape_threat
        self.shape_center = shape_center
        self.shape_opp_win = shape_opp_win
        self._qt: dict[bytes, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)

    # ── Q-table ───────────────────────────────────────────────────────────────

    def _q(self, key: bytes) -> np.ndarray:
        if key not in self._qt:
            self._qt[key] = np.array([0.0]*7 + [1.0]*7, dtype=np.float32)
        d = self._qt[key]
        return d[:7] / np.maximum(d[7:], 1.0)

    def _update(self, traj: list, g_final: float) -> None:
        seen = set()
        G = g_final
        for key, col, r_shape in reversed(traj):
            G = r_shape + (self.gamma * G)
            if (key, col) not in seen:
                seen.add((key, col))
                if key not in self._qt:
                    self._qt[key] = np.array([0.0]*7 + [1.0]*7, dtype=np.float32)
                self._qt[key][col] += G
                self._qt[key][col + 7] += 1.0

    def _canonical(self, board_pov: np.ndarray) -> tuple[bytes, bool]:
        key = _pack(board_pov)
        mirror = _pack(board_pov[:, ::-1])
        if mirror < key:
            return mirror, True
        return key, False

    def _mirror_col(self, col: int) -> int:
        return ConnectState.COLS - 1 - col

    def _eps(self, step: int, total: int) -> float:
        if total <= 1:
            return self.eps_end
        t = step / (total - 1)
        return self.eps_start * (1.0 - t) + self.eps_end * t

    def _count_window(self, window: np.ndarray, player: int) -> int:
        return int(np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1)

    def _count_threes(self, board: np.ndarray, player: int = 1) -> int:
        rows, cols = ConnectState.ROWS, ConnectState.COLS
        count = 0
        for r in range(rows):
            for c in range(cols - 3):
                count += self._count_window(board[r, c:c + 4], player)
        for c in range(cols):
            for r in range(rows - 3):
                count += self._count_window(board[r:r + 4, c], player)
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                count += self._count_window(window, player)
        for r in range(rows - 3):
            for c in range(3, cols):
                window = np.array([board[r + i, c - i] for i in range(4)])
                count += self._count_window(window, player)
        return count

    def _shape(self, board_pov: np.ndarray, col: int) -> float:
        reward = 0.0
        if col == 3:
            reward += self.shape_center
        before = self._count_threes(board_pov, 1)
        after_board = ConnectState(board_pov, 1).transition(col).board
        after = self._count_threes(after_board, 1)
        reward += (after - before) * self.shape_threat
        opp_state = ConnectState(after_board, -1)
        if not opp_state.is_final():
            for oc in opp_state.get_free_cols():
                if opp_state.is_applicable(oc) and opp_state.transition(oc).get_winner() == -1:
                    reward -= self.shape_opp_win
                    break
        return reward

    def _select_action(self, board_pov: np.ndarray, free: list, eps: float) -> tuple[int, bytes, int]:
        key, mirrored = self._canonical(board_pov)
        if self._rng.random() < eps:
            col = int(self._rng.choice(free))
        else:
            q = self._q(key)
            free_c = [self._mirror_col(c) if mirrored else c for c in free]
            best_c = max(free_c, key=lambda c: q[c])
            col = self._mirror_col(best_c) if mirrored else best_c
        col_c = self._mirror_col(col) if mirrored else col
        return col, key, col_c

    def _final_reward(self, winner: int, color: int) -> float:
        if winner == color:
            return 1.0
        if winner == 0:
            return 0.0
        return -1.0

    def _ep(self, color: int, eps: float, opp_action) -> None:
        state, traj, first = ConnectState(), [], True
        while not state.is_final():
            free = state.get_free_cols()
            if state.player == color:
                bpov = state.board * color
                if first:
                    col = int(self._rng.choice(free))
                    key, mirrored = self._canonical(bpov)
                    col_c = self._mirror_col(col) if mirrored else col
                else:
                    col, key, col_c = self._select_action(bpov, free, eps)
                first = False
                r_shape = self._shape(bpov, col)
                traj.append((key, col_c, r_shape))
            else:
                col = opp_action(state, -color)
            state = state.transition(col)
        w = state.get_winner()
        self._update(traj, self._final_reward(w, color))

    def _ep_self_play(self, eps: float) -> None:
        state, trajs = ConnectState(), {-1: [], 1: []}
        while not state.is_final():
            p = state.player
            bpov = state.board * p
            free = state.get_free_cols()
            col, key, col_c = self._select_action(bpov, free, eps)
            r_shape = self._shape(bpov, col)
            trajs[p].append((key, col_c, r_shape))
            state = state.transition(col)
        w = state.get_winner()
        for p in (-1, 1):
            self._update(trajs[p], self._final_reward(w, p))

    def _heuristic_action(self, state: ConnectState, player: int) -> int:
        free = state.get_free_cols()
        for col in free:
            if ConnectState(state.board, player).transition(col).get_winner() == player:
                return col
        for col in free:
            if ConnectState(state.board, -player).transition(col).get_winner() == -player:
                return col
        if 3 in free:
            return 3
        return int(self._rng.choice(free))

    def _random_action(self, state: ConnectState, _player: int) -> int:
        return int(self._rng.choice(state.get_free_cols()))

    def _train_loop(self, n: int, eps_start: int, total: int, opp_action) -> int:
        step = eps_start
        for i in range(n):
            eps = self._eps(step, total)
            self._ep(-1 if i % 2 == 0 else 1, eps, opp_action)
            step += 1
        return step

    # ── Policy API ────────────────────────────────────────────────────────────

    def mount(self, *args, **_kw) -> None:
        if self.cache and os.path.exists(self.cache):
            with open(self.cache, "rb") as f: self._qt = pickle.load(f)
            return
        deadline = time.monotonic() + args[0] if args and isinstance(args[0], (int, float)) else None
        done = lambda: deadline and time.monotonic() >= deadline
        total = self.n_vs_random + self.n_self_play + self.n_vs_heuristic
        step = 0
        if done(): return
        step = self._train_loop(self.n_vs_random, step, total, self._random_action)
        if done(): return
        step = self._train_loop(self.n_vs_heuristic, step, total, self._heuristic_action)
        if done(): return
        for _ in range(self.n_self_play):
            if done(): return
            eps = self._eps(step, total)
            self._ep_self_play(eps)
            step += 1
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
        key, mirrored = self._canonical(bpov)
        q = self._q(key)
        free_c = [self._mirror_col(c) if mirrored else c for c in free]
        best_c = max(free_c, key=lambda c: q[c])
        return int(self._mirror_col(best_c) if mirrored else best_c)