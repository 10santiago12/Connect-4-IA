import numpy as np
import pickle
import os
try:
    from typing import override
except ImportError:  # Python < 3.12
    try:
        from typing_extensions import override
    except ImportError:
        def override(func):
            return func
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class FVMCPolicy(Policy):
    """
    Connect-4 agent based on First-Visit Monte Carlo (FVMC) with GPI.

    - mount(): trains offline via FVMC + Exploring Starts against a random opponent.
      Optionally refines further with self-play (Alternating Markov Game style).
    - act(): greedy over learned Q(s, a); falls back to win-block-random heuristic
      for unseen states.

    Q-table key: tobytes() of the 6x7 board (fast, lossless hash of the full state).
    Reward convention (Bernoulli game, gamma=1):
        +1  = win for this agent
        -1  = loss for this agent
         0  = draw
    For self-play, the same Q-table is used for both colors; the active player's
    perspective is flipped so Q always represents "win probability for the player
    to move".
    """

    # ------------------------------------------------------------------ config
    ROWS = 6
    COLS = 7

    def __init__(
        self,
        n_episodes_vs_random: int = 60_000,
        n_episodes_self_play: int = 20_000,
        gamma: float = 1.0,
        cache_path: str | None = "fvmc_qtable.pkl",
        use_win_prob_exploration: bool = True,
    ):
        self.n_episodes_vs_random = n_episodes_vs_random
        self.n_episodes_self_play = n_episodes_self_play
        self.gamma = gamma
        self.cache_path = cache_path
        self.use_win_prob_exploration = use_win_prob_exploration

        # Q[board_bytes][col] = estimated win probability
        self.Q: dict[bytes, np.ndarray] = {}
        # N[board_bytes][col] = visit count (for incremental mean update)
        self.N: dict[bytes, np.ndarray] = {}

    # ---------------------------------------------------------------- helpers

    def _key(self, board: np.ndarray) -> bytes:
        return board.tobytes()

    def _get_q(self, board: np.ndarray) -> np.ndarray:
        """Return Q-vector for this board, initializing to 0.5 if unseen."""
        k = self._key(board)
        if k not in self.Q:
            self.Q[k] = np.full(self.COLS, 0.5)
            self.N[k] = np.zeros(self.COLS, dtype=np.int32)
        return self.Q[k]

    def _free_cols(self, board: np.ndarray) -> list[int]:
        return [c for c in range(self.COLS) if board[0, c] == 0]

    def _random_action(self, board: np.ndarray) -> int:
        return int(np.random.choice(self._free_cols(board)))

    def _greedy_action(self, board: np.ndarray) -> int:
        free = self._free_cols(board)
        q = self._get_q(board)
        return int(max(free, key=lambda c: q[c]))

    def _win_prob_action(self, board: np.ndarray) -> int:
        """Sample action proportionally to Q values (win-probability exploration)."""
        free = self._free_cols(board)
        q = self._get_q(board)
        weights = np.array([max(q[c], 1e-6) for c in free])
        weights /= weights.sum()
        return int(np.random.choice(free, p=weights))

    def _explore_action(self, board: np.ndarray) -> int:
        if self.use_win_prob_exploration:
            return self._win_prob_action(board)
        # ε-greedy fallback
        if np.random.random() < 0.15:
            return self._random_action(board)
        return self._greedy_action(board)

    def _opponent_can_win_next(self, board: np.ndarray, my_color: int, col: int) -> bool:
        """
        True if playing `col` allows the opponent to win immediately next turn.
        """
        state = ConnectState(board, my_color)
        if not state.is_applicable(col):
            return True
        next_state = state.transition(col)
        opp = -my_color
        for opp_col in next_state.get_free_cols():
            opp_state = ConnectState(next_state.board, opp)
            if not opp_state.is_applicable(opp_col):
                continue
            if opp_state.transition(opp_col).get_winner() == opp:
                return True
        return False

    # -------------------------------------------------- FVMC update from trial

    def _update_from_trial(
        self,
        trajectory: list[tuple[np.ndarray, int]],  # (board, col) pairs
        final_reward: float,                        # from the perspective of player who started
    ) -> None:
        """
        First-Visit MC update for a single trial.
        trajectory: list of (board_state, action) for a single player's turns.
        final_reward: +1 win, -1 loss, 0 draw — from that player's perspective.
        """
        visited: set[tuple[bytes, int]] = set()
        G = final_reward  # gamma=1, so return is just the final reward

        for board, col in trajectory:
            k = self._key(board)
            sa = (k, col)
            if sa in visited:
                continue  # first-visit only
            visited.add(sa)

            if k not in self.Q:
                self.Q[k] = np.full(self.COLS, 0.5)
                self.N[k] = np.zeros(self.COLS, dtype=np.int32)

            self.N[k][col] += 1
            n = self.N[k][col]
            # Incremental mean: Q += (G - Q) / n
            self.Q[k][col] += (G - self.Q[k][col]) / n

    # -------------------------------------------------- episode generators

    def _run_episode_vs_random(self, agent_color: int) -> None:
        """
        Play one full game: agent (FVMC) vs random opponent.
        agent_color: -1 (Red, moves first) or +1 (Yellow, moves second).
        Uses Exploring Starts: random first action for the agent.
        """
        state = ConnectState()
        agent_trajectory: list[tuple[np.ndarray, int]] = []
        first_move = True

        while not state.is_final():
            if state.player == agent_color:
                board_pov = state.board * agent_color
                if first_move:
                    col = self._random_action(board_pov)  # Exploring Start
                    first_move = False
                else:
                    col = self._explore_action(board_pov)
                agent_trajectory.append((board_pov.copy(), col))
            else:
                col = self._random_action(state.board)

            state = state.transition(col)

        winner = state.get_winner()
        if winner == agent_color:
            reward = 1.0
        elif winner == 0:
            reward = 0.0
        else:
            reward = -1.0

        self._update_from_trial(agent_trajectory, reward)

    def _run_episode_self_play(self) -> None:
        """
        Self-play episode (Alternating Markov Game).
        Both players share Q and improve simultaneously.
        Board is stored from current player's perspective (flipped for Yellow).
        """
        state = ConnectState()
        # trajectories indexed by player: -1 and +1
        trajectories: dict[int, list[tuple[np.ndarray, int]]] = {-1: [], 1: []}

        while not state.is_final():
            player = state.player
            # Normalize board: current player is always +1 in our Q perspective
            board_from_player_pov = state.board * player
            col = self._explore_action(board_from_player_pov)
            trajectories[player].append((board_from_player_pov.copy(), col))
            state = state.transition(col)

        winner = state.get_winner()
        for player in [-1, 1]:
            if winner == player:
                reward = 1.0
            elif winner == 0:
                reward = 0.0
            else:
                reward = -1.0
            self._update_from_trial(trajectories[player], reward)

    # ------------------------------------------------------------ Policy API

    @override
    def mount(self) -> None:
        """Train offline. Loads from cache if available, otherwise trains from scratch."""
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
                self.Q = data["Q"]
                self.N = data["N"]
            return

        # Phase 1: learn basics against random opponent (both colors)
        for i in range(self.n_episodes_vs_random):
            color = -1 if i % 2 == 0 else 1
            self._run_episode_vs_random(color)

        # Phase 2: self-play refinement (Alternating Markov Game style)
        for _ in range(self.n_episodes_self_play):
            self._run_episode_self_play()

        if self.cache_path:
            with open(self.cache_path, "wb") as f:
                pickle.dump({"Q": self.Q, "N": self.N}, f)

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Choose action greedily from Q-table.
        For unseen states, uses a fast heuristic:
          1. Win immediately if possible.
          2. Block opponent's immediate win.
          3. Prefer center columns.
        """
        free = self._free_cols(s)

        # Detect which player we are from the board balance
        red_count = np.sum(s == -1)
        yellow_count = np.sum(s == 1)
        my_color = -1 if red_count == yellow_count else 1  # Red moves first
        board_pov = s * my_color

        # Avoid moves that give opponent an immediate win
        safe = [c for c in free if not self._opponent_can_win_next(s, my_color, c)]
        candidates = safe if safe else free

        # Check if this state has been seen during training (use POV key)
        k = self._key(board_pov)
        if k in self.Q:
            return int(max(candidates, key=lambda c: self.Q[k][c]))

        # Fallback heuristic for unseen states

        # 1. Win immediately
        for col in candidates:
            state = ConnectState(s, my_color)
            next_state = state.transition(col)
            if next_state.get_winner() == my_color:
                return col

        # 2. Block opponent win
        opp = -my_color
        for col in candidates:
            state = ConnectState(s, opp)
            if state.is_applicable(col):
                next_state = state.transition(col)
                if next_state.get_winner() == opp:
                    return col

        # 3. Prefer center columns
        center_order = sorted(candidates, key=lambda c: abs(c - 3))
        return center_order[0]