import numpy as np
import chess

def fen_to_tensor(fen):
    """
    Converts a FEN string into an 8x8x12 numpy array.
    """
    board = chess.Board(fen)
    # Initialize an empty 12x8x8 matrix
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Map pieces to layer indices
    piece_to_layer = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11 # Black
    }
    
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7): # 1=Pawn, 2=Knight, etc.
            squares = board.pieces(piece_type, color)
            for square in squares:
                # Convert 0-63 index to (row, col)
                row = 7 - (square // 8)
                col = square % 8
                
                # Get the correct piece character to find the layer
                piece_char = board.piece_at(square).symbol()
                layer = piece_to_layer[piece_char]
                tensor[layer][row][col] = 1.0
                
    return tensor

def move_to_index(move_uci):
    """
    Optional: Converts a move like 'e2e4' into a unique number (0-4095).
    This helps the AI treat 'move picking' as a classification problem.
    """
    from_square = chess.SQUARE_NAMES.index(move_uci[:2])
    to_square = chess.SQUARE_NAMES.index(move_uci[2:4])
    return from_square * 64 + to_square