import chess.pgn

def extract_my_moves(pgn_file_path, my_username):
    """
    Reads a PGN file and extracts the board state and the move chosen by a specific player.
    """
    dataset = [] # Will hold tuples of (Board_FEN, Move_UCI)
    games_processed = 0
    
    # Open the PGN file
    with open(pgn_file_path) as pgn:
        while True:
            # Read one game at a time
            game = chess.pgn.read_game(pgn)
            if game is None:
                break # End of file reached
            
            # Determine which color you played this game
            headers = game.headers
            if headers.get("White") == my_username:
                my_color = chess.WHITE
            elif headers.get("Black") == my_username:
                my_color = chess.BLACK
            else:
                continue # Skip games where you aren't playing
                
            # Set up a virtual board to track the game state
            board = game.board()
            
            # Iterate through every move in the game
            for move in game.mainline_moves():
                if board.turn == my_color:
                    # It is your turn! 
                    # We save the board state (as a FEN string) and your chosen move (as UCI, e.g., 'e2e4')
                    dataset.append((board.fen(), move.uci()))
                
                # Push the move to update the board for the next player's turn
                board.push(move)
                
            games_processed += 1
            if games_processed % 100 == 0:
                print(f"Processed {games_processed} games...")
                
    print(f"Extraction complete! Found {len(dataset)} individual moves.")
    return dataset

# --- How to run it ---
if __name__ == "__main__":
    # Replace with your actual Chess.com username and the path to your downloaded PGN
    USERNAME = "SRK6776" 
    PGN_FILE = "my_games.pgn" 
    
    my_data = extract_my_moves(PGN_FILE, USERNAME)
    
    # Let's look at the first 3 data points to verify
    print("\nSample Data:")
    for i in range(3):
        print(f"Board State (FEN): {my_data[i][0]}")
        print(f"Move Played: {my_data[i][1]}\n")