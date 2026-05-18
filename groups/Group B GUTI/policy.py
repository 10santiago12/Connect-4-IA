import numpy as np
from connect4.policy import Policy


def get_current_player(board: np.ndarray) -> int:
    """
    Deduce de quién es el turno contando fichas.
    El jugador -1 (Rojo) siempre mueve primero.
    Si hay igual cantidad de fichas de cada uno → turno de -1.
    Si hay una ficha más de -1 → turno de 1.
    """
    count_red    = np.sum(board == -1)
    count_yellow = np.sum(board == 1)
    return -1 if count_red == count_yellow else 1


def normalize(board: np.ndarray, my_color: int) -> tuple:
    """
    Convierte el tablero a una tupla hashable desde la perspectiva
    del jugador actual:
      +1  → soy yo
      -1  → es el oponente
       0  → vacío
    Así la tabla Q funciona igual sin importar si soy Rojo o Amarillo.
    """
    return tuple((board * my_color).flatten())


def get_available_cols(board: np.ndarray) -> list:
    return [c for c in range(7) if board[0, c] == 0]


def apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """Aplica un movimiento y retorna el nuevo tablero."""
    new_board = board.copy()
    for r in reversed(range(6)):
        if new_board[r, col] == 0:
            new_board[r, col] = player
            break
    return new_board


def check_winner(board: np.ndarray) -> int:
    """Retorna -1 o 1 si hay ganador, 0 si no."""
    for r in range(6):
        for c in range(7):
            p = board[r, c]
            if p == 0:
                continue
            if c + 3 < 7 and all(board[r, c+i] == p for i in range(4)):
                return p
            if r + 3 < 6 and all(board[r+i, c] == p for i in range(4)):
                return p
            if r + 3 < 6 and c + 3 < 7 and all(board[r+i, c+i] == p for i in range(4)):
                return p
            if r + 3 < 6 and c - 3 >= 0 and all(board[r+i, c-i] == p for i in range(4)):
                return p
    return 0


def is_terminal(board: np.ndarray) -> bool:
    return check_winner(board) != 0 or len(get_available_cols(board)) == 0


class QLearningAgent(Policy):
    """
    Agente basado en Q-Learning tabular.

    La idea central (vista en clase):
      - Mantenemos una tabla q[(estado, accion)] que estima
        cuánto vale tomar esa accion en ese estado.
      - Despues de cada transicion actualizamos con la formula:
          q(s,a) <- q(s,a) + alpha * [r + gamma * max_a' q(s',a') - q(s,a)]
      - Esto es exactamente la actualizacion incremental de la media
        empirica, pero con un 'target' que incluye el futuro (bootstrapping).
      - La politica durante entrenamiento es epsilon-greedy:
          con prob epsilon  -> accion aleatoria  (explorar)
          con prob 1-epsilon -> argmax q(s,.)    (explotar)

    Estrategia de entrenamiento:
      - 50% de los episodios: self-play (aprende a atacar y bloquear)
      - 50% de los episodios: vs jugador aleatorio (aprende a ganar
        contra jugadas impredecibles, que es exactamente lo que evalua
        Gradescope)
    """

    # Hiperparametros
    N_EPISODES  = 40_000
    ALPHA       = 0.3
    GAMMA       = 0.95
    EPSILON     = 0.15

    # Recompensas
    R_WIN       =  1.0
    R_LOSE      = -1.0
    R_DRAW      =  0.1
    R_STEP      =  0.0

    def __init__(self):
        self.q_table = {}
        self.rng = np.random.default_rng(42)
        self._train()

    def _train(self):
        for ep in range(self.N_EPISODES):
            if ep % 2 == 0:
                self._episode_selfplay()
            else:
                self._episode_vs_random()

    def _episode_selfplay(self):
        board  = np.zeros((6, 7), dtype=int)
        player = -1
        last   = {-1: None, 1: None}

        while not is_terminal(board):
            available = get_available_cols(board)
            state_key = normalize(board, player)

            if self.rng.random() < self.EPSILON:
                col = int(self.rng.choice(available))
            else:
                col = self._best_action(state_key, available)

            new_board = apply_move(board, col, player)
            winner    = check_winner(new_board)

            if winner == player:
                self._update_q(state_key, col, self.R_WIN)
                if last[-player] is not None:
                    s, a = last[-player]
                    self._update_q(s, a, self.R_LOSE)
                break
            elif is_terminal(new_board):
                self._update_q(state_key, col, self.R_DRAW)
                if last[-player] is not None:
                    s, a = last[-player]
                    self._update_q(s, a, self.R_DRAW)
                break
            else:
                next_key   = normalize(new_board, player)
                next_avail = get_available_cols(new_board)
                max_next   = max(self.q_table.get((next_key, a), 0.0)
                                 for a in next_avail)
                self._update_q(state_key, col,
                               self.R_STEP + self.GAMMA * max_next)

            last[player] = (state_key, col)
            board  = new_board
            player = -player

    def _episode_vs_random(self):
        board       = np.zeros((6, 7), dtype=int)
        player      = -1
        agent_color = -1 if (self.rng.integers(2) == 0) else 1

        last_state = None
        last_col   = None

        while not is_terminal(board):
            available = get_available_cols(board)

            if player == agent_color:
                state_key  = normalize(board, player)
                if self.rng.random() < self.EPSILON:
                    col = int(self.rng.choice(available))
                else:
                    col = self._best_action(state_key, available)
                last_state = state_key
                last_col   = col
            else:
                col = int(self.rng.choice(available))

            new_board = apply_move(board, col, player)
            winner    = check_winner(new_board)

            if winner == agent_color:
                if last_state is not None:
                    self._update_q(last_state, last_col, self.R_WIN)
                break
            elif winner == -agent_color:
                if last_state is not None:
                    self._update_q(last_state, last_col, self.R_LOSE)
                break
            elif is_terminal(new_board):
                if last_state is not None:
                    self._update_q(last_state, last_col, self.R_DRAW)
                break
            else:
                if player == agent_color and last_state is not None:
                    next_key   = normalize(new_board, player)
                    next_avail = get_available_cols(new_board)
                    max_next   = max(self.q_table.get((next_key, a), 0.0)
                                     for a in next_avail)
                    self._update_q(last_state, last_col,
                                   self.R_STEP + self.GAMMA * max_next)

            board  = new_board
            player = -player

    def _update_q(self, state_key: tuple, col: int, target: float):
        """
        Actualizacion incremental Q-Learning:
          Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]
        """
        key     = (state_key, col)
        current = self.q_table.get(key, 0.0)
        self.q_table[key] = current + self.ALPHA * (target - current)

    def _best_action(self, state_key: tuple, available: list) -> int:
        """Retorna la columna con mayor valor Q. Si empatan, elige aleatoriamente."""
        q_values = {c: self.q_table.get((state_key, c), 0.0) for c in available}
        max_q    = max(q_values.values())
        best     = [c for c, v in q_values.items() if v == max_q]
        return int(self.rng.choice(best))

    def mount(self, time_budget=None) -> None:
        # El entrenamiento ya ocurrio en __init__.
        # mount() recibe un argumento opcional de tiempo que ignoramos.
        pass

    def act(self, s: np.ndarray) -> int:
        """
        Dado el tablero actual, deduce de quien es el turno,
        normaliza el estado y retorna la mejor columna segun Q.
        Si el estado nunca fue visto -> elige aleatoriamente entre disponibles.
        """
        available = get_available_cols(s)
        my_color  = get_current_player(s)
        state_key = normalize(s, my_color)
        return self._best_action(state_key, available)