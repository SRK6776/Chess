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

    def get_best_move(self, game_state, valid_moves):
        if not valid_moves:
            return None
            
        # 1. Digest the board state
        tensor = self.board_to_tensor(game_state.board)
        tensor = torch.tensor(tensor).unsqueeze(0).to(self.device) # Add batch dimension -> (1, 12, 8, 8)
        
        # 2. Get AI predictions
        with torch.no_grad():
            outputs = self.model(tensor)
            # Apply softmax to get an actual probability distribution array (sums to 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        # 3. Filter for strictly legal moves
        best_move = None
        highest_prob = -1.0
        
        # Build dictionary linking model index -> valid Move objects
        valid_move_dict = {self.move_to_index(m): m for m in valid_moves}
        
        # 4. Search output distribution for the valid move with the highest confidence
        for move_idx, prob in enumerate(probabilities):
            if move_idx in valid_move_dict and prob.item() > highest_prob:
                highest_prob = prob.item()
                best_move = valid_move_dict[move_idx]
                
        # Fallback (Should almost never happen if model predicts valid move space)
        if best_move is None:
            import random
            best_move = random.choice(valid_moves)
            print("AI fallback: Random move selected.")
            
        print(f"AI Confidence: {highest_prob * 100:.2f}%")
        return best_move
