import chess.pgn
import numpy as np
from features import fen_to_tensor, move_to_index

def create_training_data(pgn_file, my_username):
    X = [] # Input: Board States
    y = [] # Output: Your Moves
    
    with open(pgn_file) as pgn:
        game_count = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None: break
            
            # Determine your color
            if game.headers.get("White") == my_username:
                my_color = chess.WHITE
            elif game.headers.get("Black") == my_username:
                my_color = chess.BLACK
            else:
                continue
                
            board = game.board()
            for move in game.mainline_moves():
                if board.turn == my_color:
                    # Convert board to 12x8x8 tensor
                    X.append(fen_to_tensor(board.fen()))
                    # Convert move (e.g., 'c3c1') to a single integer index
                    y.append(move_to_index(move.uci()))
                
                board.push(move)
            
            game_count += 1

    # Convert lists to NumPy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    # Save to disk
    np.save('X.npy', X)
    np.save('y.npy', y)
    print(f"Saved {len(X)} move pairs to X.npy and y.npy")

if __name__ == "__main__":
    create_training_data("my_games.pgn", "SRK6776")