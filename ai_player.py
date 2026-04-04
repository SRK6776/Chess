import torch
import numpy as np
from train_model import ChessCNN

class AIPlayer:
    def __init__(self, model_path="chess_model.pth"):
        # Setup device and load the saved PyTorch model
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = ChessCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set model to evaluation mode
        
    def board_to_tensor(self, board_2d):
        """
        Converts the 8x8 list-of-lists board representation into the same
        12x8x8 one-hot tensor format the CNN was trained on.
        """
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_to_layer = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Black
        }
        for r in range(8):
            for c in range(8):
                piece = board_2d[r][c]
                if piece != ".":
                    layer = piece_to_layer[piece]
                    tensor[layer][r][c] = 1.0
        return tensor

    def move_to_index(self, move):
        """
        Maps the game engine's Move object to the 0-4095 continuous space
        that the model uses to output predictions.
        """
        from_sq = move.start_col + (7 - move.start_row) * 8
        to_sq = move.end_col + (7 - move.end_row) * 8
        return from_sq * 64 + to_sq

    # ─── Piece-Square Tables (from White's perspective) ─────
    # These encode positional knowledge: where each piece WANTS to be.
    # Values in centipawns (divided by 100 later). Positive = good square.
    
    PAWN_TABLE = [
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [ 5,  5, 10, 25, 25, 10,  5,  5],
        [ 0,  0,  0, 20, 20,  0,  0,  0],
        [ 5, -5,-10,  0,  0,-10, -5,  5],
        [ 5, 10, 10,-20,-20, 10, 10,  5],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ]
    
    KNIGHT_TABLE = [
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ]
    
    BISHOP_TABLE = [
        [-20,-10,-10,-10,-10,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0, 10, 10, 10, 10,  0,-10],
        [-10,  5,  5, 10, 10,  5,  5,-10],
        [-10,  0,  5, 10, 10,  5,  0,-10],
        [-10, 10, 10, 10, 10, 10, 10,-10],
        [-10,  5,  0,  0,  0,  0,  5,-10],
        [-20,-10,-10,-10,-10,-10,-10,-20]
    ]
    
    ROOK_TABLE = [
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 5, 10, 10, 10, 10, 10, 10,  5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [-5,  0,  0,  0,  0,  0,  0, -5],
        [ 0,  0,  0,  5,  5,  0,  0,  0]
    ]
    
    QUEEN_TABLE = [
        [-20,-10,-10, -5, -5,-10,-10,-20],
        [-10,  0,  0,  0,  0,  0,  0,-10],
        [-10,  0,  5,  5,  5,  5,  0,-10],
        [ -5,  0,  5,  5,  5,  5,  0, -5],
        [  0,  0,  5,  5,  5,  5,  0, -5],
        [-10,  5,  5,  5,  5,  5,  0,-10],
        [-10,  0,  5,  0,  0,  0,  0,-10],
        [-20,-10,-10, -5, -5,-10,-10,-20]
    ]
    
    KING_TABLE_MIDDLE = [
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-30,-40,-40,-50,-50,-40,-40,-30],
        [-20,-30,-30,-40,-40,-30,-30,-20],
        [-10,-20,-20,-20,-20,-20,-20,-10],
        [ 20, 20,  0,  0,  0,  0, 20, 20],
        [ 20, 30, 10,  0,  0, 10, 30, 20]
    ]

    KING_TABLE_ENDGAME = [
        [-50,-40,-30,-20,-20,-30,-40,-50],
        [-30,-20,-10,  0,  0,-10,-20,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 30, 40, 40, 30,-10,-30],
        [-30,-10, 20, 30, 30, 20,-10,-30],
        [-30,-30,  0,  0,  0,  0,-30,-30],
        [-50,-30,-30,-30,-30,-30,-30,-50]
    ]
    
    PST_MIDDLE = {
        'P': PAWN_TABLE, 'N': KNIGHT_TABLE, 'B': BISHOP_TABLE,
        'R': ROOK_TABLE, 'Q': QUEEN_TABLE, 'K': KING_TABLE_MIDDLE,
    }

    PST_ENDGAME = {
        'P': PAWN_TABLE, 'N': KNIGHT_TABLE, 'B': BISHOP_TABLE,
        'R': ROOK_TABLE, 'Q': QUEEN_TABLE, 'K': KING_TABLE_ENDGAME,
    }

    PIECE_BASE_VALUES = { 'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0 }

    def evaluate_board(self, game_state):
        """
        Material + Positional Evaluation using Piece-Square Tables.
        Dynamically detects game phase and acts appropriately.
        """
        # Determine if we are in a safe endgame where King activity is required.
        # General chess rule: If Queens are on the board, King safety is paramount.
        white_queen_alive = False
        black_queen_alive = False
        non_pawn_material = 0
        
        for r in range(8):
            for c in range(8):
                piece = game_state.board[r][c]
                if piece == 'Q': white_queen_alive = True
                if piece == 'q': black_queen_alive = True
                if piece not in ['.', 'P', 'p', 'K', 'k']:
                    non_pawn_material += self.PIECE_BASE_VALUES[piece.upper()]
                    
        # Only switch to Endgame tables (which centralize the king) IF:
        # Both queens are dead, OR one queen is dead and the other side has almost no pieces left.
        queens_dead = not white_queen_alive and not black_queen_alive
        is_endgame = queens_dead or (non_pawn_material < 1200)
        
        active_pst = self.PST_ENDGAME if is_endgame else self.PST_MIDDLE

        score = 0
        for r in range(8):
            for c in range(8):
                piece = game_state.board[r][c]
                if piece == '.':
                    continue
                    
                    
                p_type = piece.upper()
                base = self.PIECE_BASE_VALUES[p_type]
                pst_bonus = active_pst[p_type][r][c] if piece.isupper() else active_pst[p_type][7 - r][c]
                
                eval_val = base + pst_bonus
                
                # --- Advanced Positional Rules ---
                if p_type == 'P':
                    # Passed Pawn (worth almost a minor piece in endgame)
                    passed = True
                    if piece.isupper():
                        for er in range(0, r):
                            for ec in range(max(0, c-1), min(8, c+2)):
                                if game_state.board[er][ec] == 'p': passed = False
                    else:
                        for er in range(r+1, 8):
                            for ec in range(max(0, c-1), min(8, c+2)):
                                if game_state.board[er][ec] == 'P': passed = False
                    if passed: eval_val += 50
                        
                if p_type == 'R':
                    # Open File (no friendly pawns blocking)
                    own_pawn = 'P' if piece.isupper() else 'p'
                    if not any(game_state.board[tr][c] == own_pawn for tr in range(8)):
                        eval_val += 25
                        
                if piece.isupper():
                    score += eval_val
                else:
                    score -= eval_val
        
        return score / 100.0  # Convert centipawns to pawn units

    def _capture_score(self, move):
        """MVV-LVA: prioritize capturing high-value pieces with low-value attackers."""
        if move.piece_captured == '.':
            return 0
        victim = self.PIECE_BASE_VALUES.get(move.piece_captured.upper(), 0)
        attacker = self.PIECE_BASE_VALUES.get(move.piece_moved.upper(), 0)
        return victim * 10 - attacker  # High victim, low attacker = best

    def quiescence_search(self, game_state, alpha, beta, is_maximizing):
        """
        Solves the Horizon Effect by continuing search for captures only.
        """
        # Critical: Check for mate at the horizon before standing pat!
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            if game_state.is_in_check(): # Checkmate
                return -9999 if is_maximizing else 9999
            return 0 # Stalemate
            
        stand_pat = self.evaluate_board(game_state)
        
        if is_maximizing:
            if stand_pat >= beta: return beta
            if stand_pat > alpha: alpha = stand_pat
        else:
            if stand_pat <= alpha: return alpha
            if stand_pat < beta: beta = stand_pat
            
        capture_moves = [m for m in valid_moves if m.piece_captured != '.']
        capture_moves.sort(key=lambda m: self._capture_score(m), reverse=True)
        
        if is_maximizing:
            for move in capture_moves:
                game_state.make_move(move)
                score = self.quiescence_search(game_state, alpha, beta, False)
                game_state.undo_move()
                
                if score >= beta: return beta
                if score > alpha: alpha = score
            return alpha
        else:
            for move in capture_moves:
                game_state.make_move(move)
                score = self.quiescence_search(game_state, alpha, beta, True)
                game_state.undo_move()
                
                if score <= alpha: return alpha
                if score < beta: beta = score
            return beta

    def fast_minimax(self, game_state, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return self.quiescence_search(game_state, alpha, beta, is_maximizing)
            
        valid_moves = game_state.get_valid_moves()
        
        # TACTICAL END-STATE CHECK
        if not valid_moves:
            if game_state.is_in_check(): # Checkmate
                return -9999 if is_maximizing else 9999
            return 0 # Stalemate
            
        # Move Ordering: MVV-LVA for captures, then quiet moves
        valid_moves.sort(key=lambda m: self._capture_score(m), reverse=True)
        
        if is_maximizing:
            max_eval = -float('inf')
            for move in valid_moves:
                game_state.make_move(move)
                eval = self.fast_minimax(game_state, depth - 1, alpha, beta, False)
                game_state.undo_move()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                game_state.make_move(move)
                eval = self.fast_minimax(game_state, depth - 1, alpha, beta, True)
                game_state.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_best_move(self, game_state, valid_moves, depth=3):
        if not valid_moves:
            return None
            
        # 1. Ask CNN for YOUR style mapping
        tensor = self.board_to_tensor(game_state.board)
        tensor = torch.tensor(tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        move_probs = {}
        for move in valid_moves:
            idx = self.move_to_index(move)
            move_probs[move] = probabilities[idx].item()
            
        is_maximizing = game_state.white_to_move
        best_move = None
        STYLE_WEIGHT = 2.5
        
        # 2. Hybrid Tactical Calculation
        if is_maximizing:
            best_score = -float('inf')
            for move in valid_moves:
                game_state.make_move(move)
                tactical_score = self.fast_minimax(game_state, depth - 1, -float('inf'), float('inf'), False)
                game_state.undo_move()
                
                final_score = tactical_score + (move_probs[move] * STYLE_WEIGHT)
                if final_score > best_score:
                    best_score = final_score
                    best_move = move
        else:
            best_score = float('inf')
            for move in valid_moves:
                game_state.make_move(move)
                tactical_score = self.fast_minimax(game_state, depth - 1, -float('inf'), float('inf'), True)
                game_state.undo_move()
                
                final_score = tactical_score - (move_probs[move] * STYLE_WEIGHT)
                if final_score < best_score:
                    best_score = final_score
                    best_move = move

        if best_move is None:
            best_move = valid_moves[0]
            
        print(f"Hybrid AI evaluation: {best_score:.2f}")
        return best_move
