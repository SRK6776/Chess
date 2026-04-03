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

    def evaluate_board(self, game_state):
        """
        Pure Material Evaluation. 
        Does not look at flags to avoid search poisoning.
        """
        piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0, 
            '.': 0
        }
        score = 0
        for row in game_state.board:
            for piece in row:
                score += piece_values[piece]
        return score

    def fast_minimax(self, game_state, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return self.evaluate_board(game_state)
            
        valid_moves = game_state.get_valid_moves()
        
        # TACTICAL END-STATE CHECK
        if not valid_moves:
            if game_state.is_in_check(): # Checkmate
                return -9999 if is_maximizing else 9999
            return 0 # Stalemate
            
        # Move Ordering: Evaluate occupies/captures first
        valid_moves.sort(key=lambda m: 1 if m.piece_captured != "." else 0, reverse=True)
        
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
