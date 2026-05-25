import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class FlatMonteCarlo(Policy):

    def __init__(self, N: int = 200):
        # N = presupuesto de trials (rollouts) por accion en cada decision
        self.N = N
        self.rng = np.random.default_rng()

    def mount(self, timeout=None) -> None:
        # Se llama una vez antes de cada partida; reiniciamos el generador aleatorio
        self.rng = np.random.default_rng()

    def act(self, board: np.ndarray) -> int:
        # Columnas legales directamente del tablero (fila 0 libre)
        legal_cols = [c for c in range(7) if board[0, c] == 0]
        if not legal_cols:
            return 0

        # Inferimos el jugador actual: Red (-1) mueve primero.
        # Si ambos tienen igual cantidad de fichas, es turno de Red; si no, de Yellow.
        current_player = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
        state = ConnectState(board=board, player=current_player)

        # Si el estado ya es final, devolvemos la primera columna legal
        if state.is_final():
            return legal_cols[0]

        # --- Flat Monte Carlo ---
        # Para cada accion legal a, estimamos q^(s, a) como la media empirica
        # de las utilidades observadas U_hat sobre N trials (rollouts aleatorios).
        # q_hat(s, a) = (victorias en N rollouts desde (s, a)) / N
        q_hat = {}
        for col in legal_cols:
            next_state = state.transition(col)  # aplicar accion a
            wins = sum(self._rollout(next_state, current_player) for _ in range(self.N))
            q_hat[col] = wins / self.N  # q-value estimado para este par (s, a)

        # Argmax: elegir la accion con el mayor q-value estimado
        return max(q_hat, key=lambda c: q_hat[c])

    def _rollout(self, state: ConnectState, our_player: int) -> float:
        """
        Ejecuta un trial aleatorio (rollout) desde el estado dado hasta un estado terminal.

        Ambos jugadores actuan con la politica por defecto (aleatoria uniforme),
        generando un "inner trial" o "trash trial" en el vocabulario del curso.

        Utilidad observada U_hat_tau:
          1.0  si our_player gana este trial  (recompensa terminal positiva)
          0.0  si pierde o empata            (sin descuento: gamma = 1)
        """
        sim = state
        while not sim.is_final():
            col = int(self.rng.choice(sim.get_free_cols()))
            sim = sim.transition(col)
        return 1.0 if sim.get_winner() == our_player else 0.0