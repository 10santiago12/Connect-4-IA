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
    """Retorna -1, 1 si hay ganador, 0 si no."""
    for r in range(6):
        for c in range(7):
            p = board[r, c]
            if p == 0:
                continue
            # Horizontal
            if c + 3 < 7 and all(board[r, c+i] == p for i in range(4)):
                return p
            # Vertical
            if r + 3 < 6 and all(board[r+i, c] == p for i in range(4)):
                return p
            # Diagonal derecha-abajo
            if r + 3 < 6 and c + 3 < 7 and all(board[r+i, c+i] == p for i in range(4)):
                return p
            # Diagonal izquierda-abajo
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
        cuánto vale tomar esa acción en ese estado.
      - Después de cada transición actualizamos con la fórmula:
          q(s,a) ← q(s,a) + alpha * [r + gamma * max_a' q(s',a') - q(s,a)]
      - Esto es exactamente la actualización incremental de la media
        empírica, pero con un 'target' que incluye el futuro (bootstrapping).
      - La política durante entrenamiento es epsilon-greedy:
          con prob epsilon  → acción aleatoria  (explorar)
          con prob 1-epsilon → argmax q(s,·)    (explotar)
    """

    # ── Hiperparámetros ────────────────────────────────────────────────────────
    N_EPISODES  = 8_000   # partidas de entrenamiento
    ALPHA       = 0.3     # tasa de aprendizaje
    GAMMA       = 0.95    # factor de descuento
    EPSILON     = 0.2     # probabilidad de exploración

    # Recompensas
    R_WIN       =  1.0
    R_LOSE      = -1.0
    R_DRAW      =  0.0
    R_STEP      =  0.0    # movimiento neutro

    def __init__(self):
        # La tabla Q: {(estado_normalizado, col): valor_float}
        # Empieza vacía; solo se llena con lo que el agente experimenta.
        self.q_table: dict = {}
        self.rng = np.random.default_rng(42)
        self._train()

    # ── Entrenamiento (se ejecuta UNA vez al instanciar) ───────────────────────

    def _train(self):
        """
        El agente juega N_EPISODES partidas contra sí mismo (self-play).
        En cada turno aplica Q-Learning para actualizar la tabla.
        """
        for _ in range(self.N_EPISODES):
            board = np.zeros((6, 7), dtype=int)

            # Guardamos la transición del turno anterior de cada jugador
            # para poder hacer la actualización Q cuando llega la recompensa.
            # Estructura: {player: (estado_norm, col)}
            last: dict = {-1: None, 1: None}

            player = -1  # Rojo empieza siempre

            while not is_terminal(board):
                available = get_available_cols(board)
                state_key = normalize(board, player)

                # ── Política epsilon-greedy ────────────────────────────────
                if self.rng.random() < self.EPSILON:
                    col = int(self.rng.choice(available))
                else:
                    col = self._best_action(state_key, available)

                # ── Aplicar movimiento ─────────────────────────────────────
                new_board = apply_move(board, col, player)
                winner    = check_winner(new_board)

                # ── Determinar recompensa ──────────────────────────────────
                if winner == player:
                    r_current  = self.R_WIN
                    r_opponent = self.R_LOSE
                    done = True
                elif is_terminal(new_board):
                    r_current  = self.R_DRAW
                    r_opponent = self.R_DRAW
                    done = True
                else:
                    r_current  = self.R_STEP
                    r_opponent = None
                    done = False

                # ── Actualización Q para el jugador actual ─────────────────
                if done:
                    # Estado terminal: no hay futuro, target = r
                    target_current = r_current
                    self._update_q(state_key, col, target_current)

                    # También actualizamos al oponente con su recompensa
                    if last[-player] is not None:
                        opp_state, opp_col = last[-player]
                        self._update_q(opp_state, opp_col, r_opponent)
                else:
                    # Estado no terminal: guardamos (s, a) para actualizar
                    # cuando llegue la recompensa en el siguiente turno propio.
                    # Por ahora hacemos una actualización con r=0 y bootstrap.
                    next_state_key = normalize(new_board, player)
                    next_available = get_available_cols(new_board)
                    # Bootstrap: target = r + gamma * max_a' Q(s', a')
                    max_next = max(
                        self.q_table.get((next_state_key, a), 0.0)
                        for a in next_available
                    )
                    target = r_current + self.GAMMA * max_next
                    self._update_q(state_key, col, target)

                last[player] = (state_key, col)
                board  = new_board
                player = -player  # cambiar turno

    def _update_q(self, state_key: tuple, col: int, target: float):
        """
        Actualización incremental Q-Learning:
          Q(s,a) ← Q(s,a) + alpha * [target - Q(s,a)]
        Equivalente a la fórmula de media empírica incremental del curso,
        pero con un target que incluye recompensa futura (bootstrapping).
        """
        key = (state_key, col)
        current = self.q_table.get(key, 0.0)
        self.q_table[key] = current + self.ALPHA * (target - current)

    def _best_action(self, state_key: tuple, available: list) -> int:
        """Retorna la columna con mayor valor Q. Si empatan, elige aleatoriamente."""
        q_values = {c: self.q_table.get((state_key, c), 0.0) for c in available}
        max_q    = max(q_values.values())
        best     = [c for c, v in q_values.items() if v == max_q]
        return int(self.rng.choice(best))

    # ── Interfaz requerida por Policy ──────────────────────────────────────────

    def mount(self) -> None:
        # El entrenamiento ya ocurrió en __init__.
        # mount() se llama antes de cada partida, no hacemos nada aquí.
        pass

    def act(self, s: np.ndarray) -> int:
        """
        Dado el tablero actual, deduce de quién es el turno,
        normaliza el estado y retorna la mejor columna según Q.
        Si el estado nunca fue visto → elige aleatoriamente.
        """
        available  = get_available_cols(s)
        my_color   = get_current_player(s)
        state_key  = normalize(s, my_color)
        return self._best_action(state_key, available)