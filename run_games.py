"""
Corre N partidas individuales entre dos grupos y muestra los resultados.
Uso: python run_games.py
"""

from connect4.connect_state import ConnectState
from connect4.utils import find_importable_classes
from connect4.policy import Policy

# ── Configuracion ──────────────────────────────────────────────────────────────
PLAYER_A   = "Group C SANTI"   # CARPETA DEL AGENTE 1
PLAYER_B   = "Group D RANDOM"   # CARPETA DEL AGENTE 2
N_GAMES    = 10
# ──────────────────────────────────────────────────────────────────────────────

SYMBOLS = {-1: "R", 1: "Y", 0: "."}


def render_board(board) -> str:
    rows = []
    for r in range(board.shape[0]):
        rows.append(" ".join(SYMBOLS[board[r, c]] for c in range(board.shape[1])))
    rows.append("0 1 2 3 4 5 6")
    return "\n".join(rows)


def play_game(red_name, red_policy, yellow_name, yellow_policy) -> str:
    """Juega una partida. Red = jugador -1, Yellow = jugador 1. Retorna nombre del ganador."""
    red_policy.mount()
    yellow_policy.mount()

    state = ConnectState()
    while not state.is_final():
        current = red_policy if state.player == -1 else yellow_policy
        action  = current.act(state.board)
        state   = state.transition(int(action))

    winner = state.get_winner()
    if winner == -1:
        return red_name, state.board
    elif winner == 1:
        return yellow_name, state.board
    else:
        return "Draw", state.board


def main():
    participants = find_importable_classes("groups", Policy)

    if PLAYER_A not in participants:
        raise ValueError(f"No se encontro '{PLAYER_A}'. Disponibles: {list(participants)}")
    if PLAYER_B not in participants:
        raise ValueError(f"No se encontro '{PLAYER_B}'. Disponibles: {list(participants)}")

    class_a = participants[PLAYER_A]
    class_b = participants[PLAYER_B]

    wins   = {PLAYER_A: 0, PLAYER_B: 0, "Draw": 0}
    header = f"{'Juego':>6}  {'Red (-1)':^12}  {'Yellow (1)':^12}  {'Ganador':^12}"
    print(header)
    print("-" * len(header))

    for i in range(N_GAMES):
        # Alternamos quien juega de rojo para que sea justo
        if i % 2 == 0:
            red_name, red_pol     = PLAYER_A, class_a()
            yellow_name, yel_pol  = PLAYER_B, class_b()
        else:
            red_name, red_pol     = PLAYER_B, class_b()
            yellow_name, yel_pol  = PLAYER_A, class_a()

        winner, board = play_game(red_name, red_pol, yellow_name, yel_pol)
        wins[winner] += 1

        print(f"{i+1:>6}  {red_name:^12}  {yellow_name:^12}  {winner:^12}")
        print(render_board(board))
        print()

    print("=" * len(header))
    print(f"  {PLAYER_A}: {wins[PLAYER_A]} victorias")
    print(f"  {PLAYER_B}: {wins[PLAYER_B]} victorias")
    print(f"  Empates:  {wins['Draw']}")


if __name__ == "__main__":
    main()