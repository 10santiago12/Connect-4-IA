"""
run_games.py — Connect-4 con interfaz gráfica, reloj en vivo y pausa entre partidas.
Uso: python run_games.py
"""

import time
import queue
import threading
import concurrent.futures
import numpy as np
import tkinter as tk
from tkinter import ttk

from connect4.connect_state import ConnectState
from connect4.utils import find_importable_classes
from connect4.policy import Policy

# ── Paleta ─────────────────────────────────────────────────────────────────────
BG           = "#1a1a2e"
PANEL_BG     = "#16213e"
ACCENT       = "#0f3460"
BTN_START    = "#27ae60"
BTN_STOP     = "#e74c3c"
BTN_NEXT     = "#2980b9"
BTN_FG       = "white"
BOARD_BG     = "#1450a3"
BOARD_FRAME  = "#0b3280"
SHADOW       = "#071d5c"
CELL_EMPTY   = "#0b3a8a"
CELL_EMPIN   = "#0e4db5"   # anillo interior celda vacía (efecto cóncavo)
CELL_RED     = "#c62828"
CELL_RED_HI  = "#ff8a80"   # destello rojo
CELL_YEL     = "#f9a825"
CELL_YEL_HI  = "#fffde7"   # destello amarillo
WIN_OUTLINE  = "#ffffff"
TEXT_FG      = "#ecf0f1"
DIM_FG       = "#7f8c8d"
GOLD         = "#f1c40f"

CELL_SIZE = 74
RADIUS    = 30
ROWS      = 6
COLS      = 7
BOARD_W   = COLS * CELL_SIZE
BOARD_H   = ROWS * CELL_SIZE
DROP_MS   = 52   # ms por frame de animación de caída


class App(tk.Tk):
    def __init__(self, participants: dict):
        super().__init__()
        self.participants = participants
        self.names = sorted(participants.keys())

        self.title("Connect-4  ·  Battle Arena")
        self.configure(bg=BG)
        self.resizable(False, False)

        # Cola de comunicación hilo ↔ UI
        self._q             = queue.Queue()
        self._running       = False
        self._stop_flag     = threading.Event()
        self._next_evt      = threading.Event()   # pausa entre partidas
        self._wins: dict    = {}

        # Estado del reloj en vivo
        self._think_player    = None   # jugador pensando (-1 / 1 / None)
        self._think_remaining = 0.0
        self._think_start     = 0.0

        # Estado del tablero mostrado
        self._shown_board = np.zeros((ROWS, COLS), dtype=int)
        self._last_rc     = None   # (row, col) del último movimiento
        self._animating   = False

        self._apply_style()
        self._build_ui()
        self._tick_clocks()   # arranca el loop de reloj en vivo
        self._poll_queue()

    # ── Estilo ttk ──────────────────────────────────────────────────────────────

    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TCombobox",
                     fieldbackground=ACCENT, background=ACCENT,
                     foreground=TEXT_FG, selectbackground=ACCENT,
                     selectforeground=TEXT_FG, arrowcolor=TEXT_FG)
        s.map("TCombobox",
              fieldbackground=[("readonly", ACCENT)],
              foreground=[("readonly", TEXT_FG)])

    # ── Construcción de UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        left = tk.Frame(self, bg=PANEL_BG, padx=18, pady=18)
        left.pack(side="left", fill="y")
        right = tk.Frame(self, bg=BG, padx=18, pady=18)
        right.pack(side="right", fill="both", expand=True)
        self._build_left(left)
        self._build_right(right)

    # ── Panel izquierdo ─────────────────────────────────────────────────────────

    def _build_left(self, p):
        self._lbl(p, "⚙  Configuración", GOLD, 13, "bold").pack(anchor="w", pady=(0, 14))
        self._player_a_var = self._player_row(p, "Jugador  R", CELL_RED, 0)
        self._player_b_var = self._player_row(p, "Jugador  Y", CELL_YEL, 1)
        tk.Frame(p, bg=PANEL_BG, height=10).pack()
        self._games_var = self._spinner_row(p, "Partidas:", 1, 100, 10)
        self._time_var  = self._spinner_row(p, "Tiempo (s):", 5, 600, 60)
        tk.Frame(p, bg=PANEL_BG, height=14).pack()

        self._btn_start = tk.Button(
            p, text="▶  INICIAR", bg=BTN_START, fg=BTN_FG,
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
            padx=10, pady=7, command=self._start_match)
        self._btn_start.pack(fill="x", pady=3)

        self._btn_stop = tk.Button(
            p, text="⏹  DETENER", bg=BTN_STOP, fg=BTN_FG,
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2",
            padx=10, pady=7, state="disabled", command=self._stop_match)
        self._btn_stop.pack(fill="x", pady=3)

        tk.Frame(p, bg=PANEL_BG, height=14).pack()
        self._lbl(p, "📊  Marcador", GOLD, 11, "bold").pack(anchor="w")
        self._score_frame = tk.Frame(p, bg=PANEL_BG)
        self._score_frame.pack(fill="x", pady=6)
        self._refresh_score()
        tk.Frame(p, bg=PANEL_BG, height=10).pack()
        self._lbl(p, "📋  Historial", GOLD, 11, "bold").pack(anchor="w")
        self._log = tk.Text(p, bg=ACCENT, fg=TEXT_FG, width=28, height=14,
                             relief="flat", state="disabled",
                             font=("Consolas", 9), wrap="none")
        self._log.pack(fill="x", pady=4)

    def _player_row(self, parent, label, color, idx) -> tk.StringVar:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=5)
        tk.Label(row, text="●", bg=PANEL_BG, fg=color,
                 font=("Segoe UI", 14)).pack(side="left")
        tk.Label(row, text=f" {label}", bg=PANEL_BG, fg=TEXT_FG,
                 font=("Segoe UI", 10), width=11, anchor="w").pack(side="left")
        default = self.names[idx % len(self.names)] if self.names else ""
        var = tk.StringVar(value=default)
        ttk.Combobox(row, textvariable=var, values=self.names,
                     state="readonly", width=14).pack(side="left")
        return var

    def _spinner_row(self, parent, label, lo, hi, default) -> tk.IntVar:
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=4)
        tk.Label(row, text=label, bg=PANEL_BG, fg=TEXT_FG,
                 font=("Segoe UI", 10), width=13, anchor="w").pack(side="left")
        var = tk.IntVar(value=default)
        tk.Spinbox(row, from_=lo, to=hi, textvariable=var, width=7,
                   bg=ACCENT, fg=TEXT_FG, insertbackground=TEXT_FG,
                   buttonbackground=ACCENT, relief="flat").pack(side="left")
        return var

    def _lbl(self, parent, text, fg, size, weight="normal") -> tk.Label:
        return tk.Label(parent, text=text, bg=PANEL_BG, fg=fg,
                        font=("Segoe UI", size, weight))

    # ── Panel derecho ───────────────────────────────────────────────────────────

    def _build_right(self, p):
        self._title_lbl = tk.Label(
            p, text="Selecciona jugadores y presiona  ▶  INICIAR",
            bg=BG, fg=DIM_FG, font=("Segoe UI", 11))
        self._title_lbl.pack(pady=(0, 6))

        clocks = tk.Frame(p, bg=BG)
        clocks.pack(pady=3)
        self._clock_r = tk.Label(clocks, text="● R  --:--.--",
                                  bg=BG, fg=CELL_RED,
                                  font=("Courier New", 16, "bold"))
        self._clock_r.pack(side="left", padx=22)
        self._clock_y = tk.Label(clocks, text="● Y  --:--.--",
                                  bg=BG, fg=CELL_YEL,
                                  font=("Courier New", 16, "bold"))
        self._clock_y.pack(side="left", padx=22)

        self._status_lbl = tk.Label(p, text="", bg=BG, fg=TEXT_FG,
                                     font=("Segoe UI", 10))
        self._status_lbl.pack(pady=4)

        board_outer = tk.Frame(p, bg=BOARD_FRAME, padx=8, pady=8)
        board_outer.pack()
        self._canvas = tk.Canvas(board_outer, width=BOARD_W, height=BOARD_H,
                                  bg=BOARD_BG, highlightthickness=0)
        self._canvas.pack()

        num_c = tk.Canvas(board_outer, width=BOARD_W, height=22,
                           bg=BOARD_FRAME, highlightthickness=0)
        num_c.pack()
        for c in range(COLS):
            x = c * CELL_SIZE + CELL_SIZE // 2
            num_c.create_text(x, 11, text=str(c),
                               fill=TEXT_FG, font=("Segoe UI", 9))

        self._game_lbl = tk.Label(p, text="", bg=BG, fg=DIM_FG,
                                   font=("Segoe UI", 9))
        self._game_lbl.pack(pady=(6, 2))

        # Botón "Siguiente Partida" — se muestra/oculta dinámicamente
        self._btn_next = tk.Button(
            p, text="▶  Siguiente Partida", bg=BTN_NEXT, fg=BTN_FG,
            font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2",
            padx=14, pady=9, command=self._next_clicked)
        # No se empaqueta aquí; aparece solo al terminar cada partida

        self._draw_empty_board()

    # ── Dibujo de tablero realista ──────────────────────────────────────────────

    def _cell_xy(self, row, col):
        return col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2

    def _put_piece(self, x, y, player: int, is_last=False, win=False):
        """Dibuja una ficha 3-D con sombra, color base y destello."""
        r     = RADIUS
        color = CELL_RED   if player == -1 else CELL_YEL
        hi    = CELL_RED_HI if player == -1 else CELL_YEL_HI
        # sombra
        self._canvas.create_oval(x-r+2, y-r+3, x+r+2, y+r+3,
                                  fill=SHADOW, outline="")
        # ficha
        outline = WIN_OUTLINE if win else ""
        ow = 3 if win else 0
        self._canvas.create_oval(x-r, y-r, x+r, y+r,
                                  fill=color, outline=outline, width=ow)
        # destello (esquina superior-izquierda)
        hr = int(r * 0.38)
        self._canvas.create_oval(x-r+6, y-r+6,
                                  x-r+6+hr*2, y-r+6+hr*2,
                                  fill=hi, outline="")
        # punto blanco en último movimiento
        if is_last:
            dr = int(r * 0.18)
            self._canvas.create_oval(x-dr, y-dr, x+dr, y+dr,
                                      fill="white", outline="")

    def _put_empty(self, x, y):
        """Celda vacía con efecto cóncavo (dos círculos concéntricos)."""
        r  = RADIUS
        ir = int(r * 0.82)
        self._canvas.create_oval(x-r, y-r, x+r, y+r,
                                  fill=CELL_EMPTY, outline="")
        self._canvas.create_oval(x-ir, y-ir, x+ir, y+ir,
                                  fill=CELL_EMPIN, outline="")

    def _draw_empty_board(self):
        self._canvas.delete("all")
        self._shown_board = np.zeros((ROWS, COLS), dtype=int)
        self._last_rc = None
        for r in range(ROWS):
            for c in range(COLS):
                self._put_empty(*self._cell_xy(r, c))

    def _render_board(self, board, win_cells=None):
        """Dibuja el tablero completo sin modificar estado interno."""
        win_set = set(map(tuple, win_cells)) if win_cells else set()
        self._canvas.delete("all")
        for r in range(ROWS):
            for c in range(COLS):
                v = board[r, c]
                x, y = self._cell_xy(r, c)
                if v == 0:
                    self._put_empty(x, y)
                else:
                    self._put_piece(x, y, v,
                                    is_last=(self._last_rc == (r, c)),
                                    win=(r, c) in win_set)

    def _draw_board(self, board, win_cells=None):
        """Dibuja y actualiza _shown_board."""
        self._render_board(board, win_cells)
        self._shown_board = board.copy()

    def _winning_cells(self, board):
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r, c]
                if p == 0:
                    continue
                for dr, dc in dirs:
                    cells = [(r+i*dr, c+i*dc) for i in range(4)]
                    if all(0 <= cr < ROWS and 0 <= cc < COLS and board[cr, cc] == p
                           for cr, cc in cells):
                        return cells
        return None

    # ── Animación de caída ──────────────────────────────────────────────────────

    def _animate_drop(self, col: int, player: int, board_after, on_done):
        """Anima la ficha cayendo por la columna antes de mostrar el estado final."""
        prev = self._shown_board
        diffs = np.where(board_after[:, col] != prev[:, col])[0]
        if len(diffs) == 0:
            self._draw_board(board_after)
            on_done()
            return

        target_row = int(diffs[0])
        prev_snap  = prev.copy()   # captura antes de que _render_board cambie estado
        color = CELL_RED   if player == -1 else CELL_YEL
        hi    = CELL_RED_HI if player == -1 else CELL_YEL_HI
        r     = RADIUS

        def step(cur):
            # dibuja tablero anterior + ficha cayendo
            self._last_rc = None          # sin indicador de "último" durante animación
            self._render_board(prev_snap)
            x, y = self._cell_xy(cur, col)
            self._canvas.create_oval(x-r+2, y-r+3, x+r+2, y+r+3,
                                      fill=SHADOW, outline="")
            self._canvas.create_oval(x-r, y-r, x+r, y+r,
                                      fill=color, outline="")
            hr = int(r * 0.38)
            self._canvas.create_oval(x-r+6, y-r+6,
                                      x-r+6+hr*2, y-r+6+hr*2,
                                      fill=hi, outline="")
            if cur < target_row:
                self.after(DROP_MS, lambda: step(cur + 1))
            else:
                self._last_rc  = (target_row, col)
                self._animating = False
                self._draw_board(board_after)
                on_done()

        self._animating = True
        step(0)

    # ── Marcador y log ──────────────────────────────────────────────────────────

    def _refresh_score(self):
        for w in self._score_frame.winfo_children():
            w.destroy()
        if not self._wins:
            tk.Label(self._score_frame, text="—", bg=PANEL_BG, fg=DIM_FG,
                     font=("Segoe UI", 9)).pack(anchor="w")
            return
        for name, count in sorted(self._wins.items(), key=lambda x: -x[1]):
            fg  = DIM_FG if name == "Draw" else TEXT_FG
            row = tk.Frame(self._score_frame, bg=PANEL_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, bg=PANEL_BG, fg=fg,
                     font=("Segoe UI", 10), width=15, anchor="w").pack(side="left")
            tk.Label(row, text=str(count), bg=PANEL_BG, fg=GOLD,
                     font=("Segoe UI", 10, "bold")).pack(side="left")

    def _log_line(self, text: str):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n")
        self._log.see("end")
        self._log.configure(state="disabled")

    # ── Reloj en vivo (100 ms) ──────────────────────────────────────────────────

    def _tick_clocks(self):
        if self._think_player is not None:
            elapsed  = time.perf_counter() - self._think_start
            current  = max(0.0, self._think_remaining - elapsed)
            if self._think_player == -1:
                self._clock_r.config(text=f"● R  {self._fmt(current)}")
            else:
                self._clock_y.config(text=f"● Y  {self._fmt(current)}")
        self.after(100, self._tick_clocks)

    def _fmt(self, s: float) -> str:
        s = max(0.0, s)
        return f"{int(s//60):02d}:{s%60:05.2f}"

    def _set_clocks(self, remaining: dict):
        self._clock_r.config(text=f"● R  {self._fmt(remaining[-1])}")
        self._clock_y.config(text=f"● Y  {self._fmt(remaining[1])}")

    # ── Botón "Siguiente Partida" ───────────────────────────────────────────────

    def _show_next_btn(self, label: str):
        self._btn_next.config(text=label)
        self._btn_next.pack(pady=10)

    def _hide_next_btn(self):
        self._btn_next.pack_forget()

    def _next_clicked(self):
        self._hide_next_btn()
        self._next_evt.set()

    # ── Control general ─────────────────────────────────────────────────────────

    def _start_match(self):
        if self._running:
            return
        name_a   = self._player_a_var.get()
        name_b   = self._player_b_var.get()
        n_games  = self._games_var.get()
        time_bgt = self._time_var.get()

        self._wins = {name_a: 0, name_b: 0, "Draw": 0}
        self._refresh_score()
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._draw_empty_board()
        self._hide_next_btn()
        self._think_player = None

        self._running = True
        self._stop_flag.clear()
        self._next_evt.clear()
        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")

        threading.Thread(
            target=self._match_loop,
            args=(name_a, name_b, n_games, float(time_bgt)),
            daemon=True
        ).start()

    def _stop_match(self):
        self._stop_flag.set()
        self._next_evt.set()   # desbloquea hilo si estaba esperando

    # ── Hilo de la serie ────────────────────────────────────────────────────────

    def _match_loop(self, name_a: str, name_b: str, n_games: int, budget: float):
        cls_a = self.participants[name_a]
        cls_b = self.participants[name_b]

        for i in range(n_games):
            if self._stop_flag.is_set():
                break

            if i % 2 == 0:
                red_name, red_pol  = name_a, cls_a()
                yel_name, yel_pol  = name_b, cls_b()
            else:
                red_name, red_pol  = name_b, cls_b()
                yel_name, yel_pol  = name_a, cls_a()

            winner, used, motivo = self._play_game(
                i + 1, n_games,
                red_name, red_pol,
                yel_name, yel_pol,
                budget
            )

            if motivo == "stopped":
                break

            self._q.put(("game_result", {
                "game": i+1, "red": red_name, "yellow": yel_name,
                "winner": winner, "motivo": motivo, "used": used,
            }))

            # Pausa hasta que el usuario presione "Siguiente Partida"
            if i < n_games - 1 and not self._stop_flag.is_set():
                self._q.put(("await_next", {
                    "next": i + 2, "total": n_games,
                }))
                self._next_evt.wait()
                self._next_evt.clear()
                if self._stop_flag.is_set():
                    break

        self._q.put(("match_done", None))

    def _play_game(self, game_num, total_games,
                   red_name, red_pol, yel_name, yel_pol, budget):
        red_pol.mount()
        yel_pol.mount()

        remaining = {-1: budget, 1: budget}
        used      = {-1: 0.0,   1: 0.0}
        names     = {-1: red_name, 1: yel_name}
        state     = ConnectState()

        self._q.put(("game_start", {
            "game": game_num, "total": total_games,
            "red": red_name, "yellow": yel_name,
            "remaining": dict(remaining), "board": state.board.copy(),
        }))

        while not state.is_final():
            if self._stop_flag.is_set():
                return "—", used, "stopped"

            p      = state.player
            policy = red_pol if p == -1 else yel_pol

            self._q.put(("thinking", {
                "player": p, "name": names[p],
                "remaining": dict(remaining),
            }))

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    t0  = time.perf_counter()
                    fut = ex.submit(policy.act, state.board)
                    action = fut.result(timeout=remaining[p])
                elapsed = time.perf_counter() - t0
            except concurrent.futures.TimeoutError:
                winner = names[-p]
                self._q.put(("timeout", {
                    "loser": names[p], "winner": winner,
                    "board": state.board.copy(), "remaining": dict(remaining),
                }))
                return winner, used, "timeout"

            remaining[p] -= elapsed
            used[p]      += elapsed
            state = state.transition(int(action))

            self._q.put(("move", {
                "player": p, "name": names[p], "col": int(action),
                "elapsed": elapsed, "remaining": dict(remaining),
                "board": state.board.copy(),
            }))

            time.sleep(0.48)   # >= animación máxima (ROWS * DROP_MS = 312 ms)

        wid    = state.get_winner()
        winner = red_name if wid == -1 else (yel_name if wid == 1 else "Draw")

        self._q.put(("game_end", {
            "winner": winner, "board": state.board.copy(),
            "remaining": dict(remaining),
        }))
        return winner, used, "normal"

    # ── Cola de mensajes ────────────────────────────────────────────────────────

    def _poll_queue(self):
        if not self._animating:
            try:
                while True:
                    self._handle(*self._q.get_nowait())
            except queue.Empty:
                pass
        self.after(40, self._poll_queue)

    def _handle(self, msg: str, data):
        if msg == "game_start":
            self._think_player = None
            self._last_rc      = None
            self._title_lbl.config(
                text=f"Partida {data['game']} / {data['total']}  ·  "
                     f"R {data['red']}  vs  Y {data['yellow']}", fg=TEXT_FG)
            self._game_lbl.config(
                text=f"Partida {data['game']} de {data['total']}", fg=DIM_FG)
            self._status_lbl.config(text="Iniciando...", fg=DIM_FG)
            self._draw_board(data["board"])
            self._set_clocks(data["remaining"])

        elif msg == "thinking":
            # Arranca el reloj en vivo para este jugador
            self._think_player    = data["player"]
            self._think_remaining = data["remaining"][data["player"]]
            self._think_start     = time.perf_counter()
            color = CELL_RED if data["player"] == -1 else CELL_YEL
            self._status_lbl.config(
                text=f"Pensando:  {data['name']} ...", fg=color)

        elif msg == "move":
            # Para el reloj y anima la caída
            self._think_player = None
            self._set_clocks(data["remaining"])
            color = CELL_RED if data["player"] == -1 else CELL_YEL
            self._status_lbl.config(
                text=f"{data['name']}  jugó col {data['col']}  ({data['elapsed']:.2f}s)",
                fg=color)
            self._animate_drop(data["col"], data["player"], data["board"],
                                on_done=lambda: None)

        elif msg == "timeout":
            self._think_player = None
            self._status_lbl.config(
                text=f"⏰  {data['loser']} agotó su tiempo — gana {data['winner']}",
                fg=BTN_STOP)
            self._draw_board(data["board"])

        elif msg == "game_end":
            self._think_player = None
            self._set_clocks(data["remaining"])
            w  = data["winner"]
            fg = GOLD if w != "Draw" else DIM_FG
            self._status_lbl.config(text=f"🏆  Ganador: {w}", fg=fg)
            wc = self._winning_cells(data["board"]) if w != "Draw" else None
            self._draw_board(data["board"], win_cells=wc)

        elif msg == "game_result":
            self._wins[data["winner"]] = self._wins.get(data["winner"], 0) + 1
            self._refresh_score()
            sfx = " (tiempo)" if data["motivo"] == "timeout" else ""
            self._log_line(
                f"#{data['game']:>2}  R:{data['red']:<12}  "
                f"Y:{data['yellow']:<12}  →  {data['winner']}{sfx}"
            )

        elif msg == "await_next":
            if self._running:
                self._show_next_btn(
                    f"▶  Siguiente Partida  ({data['next']} / {data['total']})")

        elif msg == "match_done":
            self._think_player = None
            self._running = False
            self._btn_start.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._hide_next_btn()
            self._status_lbl.config(text="✔  Serie terminada", fg=BTN_START)
            self._title_lbl.config(text="Connect-4  ·  Serie completada", fg=GOLD)


def main():
    participants = find_importable_classes("groups", Policy)
    if not participants:
        print("No se encontraron agentes en la carpeta groups/")
        return
    App(participants).mainloop()


if __name__ == "__main__":
    main()