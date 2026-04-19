"""
UCI Engine Bridge — Connects the PyGame GUI to the Rust chess engine.

This module spawns the Rust binary as a subprocess and communicates
via the UCI protocol (stdin/stdout text commands). It replaces the
Python AIPlayer class with zero changes to the game engine logic.

Usage:
    from uci_bridge import UCIEngine
    engine = UCIEngine("chess-engine-rs/target/release/chess-engine-rs")
    best_move_uci = engine.go(game_state, depth=5)  # Returns "e2e4"
"""

import subprocess
import os
import sys
from engine import GameState, Move


class UCIEngine:
    def __init__(self, engine_path, depth=5):
        """Launches the Rust engine as a subprocess."""
        self.depth = depth
        self.last_eval = None  # Last evaluation in centipawns
        self.mode = "unknown"  # "hybrid" or "tactical"
        
        if not os.path.exists(engine_path):
            print(f"Error: Engine binary not found at {engine_path}")
            print("Build it with: cd chess-engine-rs && cargo build --release")
            sys.exit(1)
        
        self.process = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Initialize UCI handshake
        self._send("uci")
        self._read_until("uciok")
        
        self._send("isready")
        self._read_until("readyok")
        
        print("UCI Engine connected successfully")
    
    def _send(self, command):
        """Sends a command to the engine via stdin."""
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
    
    def _read_line(self):
        """Reads one line from the engine's stdout."""
        line = self.process.stdout.readline().strip()
        return line
    
    def _read_until(self, target):
        """Reads lines until we see one starting with target. Returns all lines."""
        lines = []
        while True:
            line = self._read_line()
            lines.append(line)
            if line.startswith(target):
                return lines
    
    def _game_state_to_uci_position(self, game_state):
        """Converts the Python GameState's move log into a UCI position string."""
        if not game_state.move_log:
            return "position startpos"
        
        moves_str = " ".join(
            move.get_chess_notation() for move in game_state.move_log
        )
        return f"position startpos moves {moves_str}"
    
    def _uci_to_move(self, uci_str, game_state, valid_moves):
        """Converts a UCI move string like 'e2e4' to the engine's Move object."""
        if len(uci_str) < 4:
            return None
        
        # Parse UCI notation
        files = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        
        start_col = files.get(uci_str[0])
        start_row = 8 - int(uci_str[1])
        end_col = files.get(uci_str[2])
        end_row = 8 - int(uci_str[3])
        
        if start_col is None or end_col is None:
            return None
        
        # Promotion piece (e.g., "e7e8q")
        promotion = None
        if len(uci_str) == 5:
            promo_char = uci_str[4].upper()
            promotion = promo_char
        
        # Find the matching Move object from valid_moves
        for move in valid_moves:
            if (move.start_row == start_row and move.start_col == start_col and
                move.end_row == end_row and move.end_col == end_col):
                if promotion and move.is_promotion_move:
                    move.promotion_choice = promotion
                return move
        
        return None
    
    def get_best_move(self, game_state, valid_moves, depth=None):
        """
        Drop-in replacement for AIPlayer.get_best_move().
        
        Sends the position to the Rust engine, waits for bestmove,
        and returns a Move object compatible with the Python game engine.
        """
        if not valid_moves:
            return None
        
        search_depth = depth if depth is not None else self.depth
        
        # Send current position
        pos_cmd = self._game_state_to_uci_position(game_state)
        self._send(pos_cmd)
        
        # Start search
        self._send(f"go depth {search_depth}")
        
        # Read response lines until we get bestmove
        best_move_uci = None
        while True:
            line = self._read_line()
            
            if line.startswith("info"):
                # Parse evaluation from info line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "cp" and i + 1 < len(parts):
                        try:
                            self.last_eval = int(parts[i + 1]) / 100.0
                        except ValueError:
                            pass
                    if part.startswith("mode="):
                        self.mode = part.split("=")[1]
            
            if line.startswith("bestmove"):
                best_move_uci = line.split()[1]
                break
        
        if not best_move_uci or best_move_uci == "0000":
            return valid_moves[0] if valid_moves else None
        
        # Convert UCI string back to a Move object
        move = self._uci_to_move(best_move_uci, game_state, valid_moves)
        
        if move is None:
            print(f"Warning: Could not map UCI move '{best_move_uci}' to valid moves")
            return valid_moves[0]
        
        eval_str = f"{self.last_eval:+.2f}" if self.last_eval is not None else "?"
        print(f"Rust engine ({self.mode}): {best_move_uci} eval={eval_str}")
        
        return move
    
    def new_game(self):
        """Resets the engine's internal state for a new game."""
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
        self.last_eval = None
    
    def quit(self):
        """Cleanly shuts down the engine subprocess."""
        try:
            self._send("quit")
            self.process.wait(timeout=2)
        except Exception:
            self.process.kill()
    
    def __del__(self):
        self.quit()
