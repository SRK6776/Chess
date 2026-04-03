from engine import GameState, Move
from ai_player import AIPlayer
import os

def print_board(board_2d):
    print("\n   a b c d e f g h")
    print("  -----------------")
    for r in range(8):
        row_str = f"{8 - r} |"
        for c in range(8):
            row_str += board_2d[r][c] + " "
        print(row_str + f"| {8 - r}")
    print("  -----------------")
    print("   a b c d e f g h\n")

def main():
    game_state = GameState()
    
    # Initialize your new PyTorch CNN AI
    if os.path.exists("chess_model.pth"):
        ai = AIPlayer("chess_model.pth")
    else:
        print("Model not trained yet! Go run train_model.py")
        return

    # Assign Player Colors (True = White, False = Black)
    player_white = True
    
    print("Welcome to AI Chess!")
    print("Type your moves in UCI format (e.g. e2e4, b1c3).")
    
    while True:
        print_board(game_state.board)
        valid_moves = game_state.get_valid_moves()
        
        if game_state.checkmate:
            winner = "Black" if game_state.white_to_move else "White"
            print(f"Checkmate! {winner} wins!")
            break
        elif game_state.stalemate:
            print("Stalemate! Game is a draw.")
            break
            
        is_human_turn = (game_state.white_to_move and player_white) or \
                        (not game_state.white_to_move and not player_white)
        
        if is_human_turn:
            move_str = input("Your move (or 'z' to undo): ").strip().lower()
            
            if move_str in ['z', 'undo']:
                if len(game_state.move_log) >= 2:
                    game_state.undo_move() # Undo AI move
                    game_state.undo_move() # Undo Human move
                elif len(game_state.move_log) == 1: # If only one move made
                    game_state.undo_move()
                else:
                    print("Nothing to undo.")
                continue
            
            # Match user input against all legal moves
            user_move = None
            for move in valid_moves:
                if move.get_chess_notation() == move_str:
                    user_move = move
                    break
            
            # Simple handling for pawn promotion choice
            if user_move and user_move.is_promotion_move:
                choice = input("Promote to (Q, R, B, N): ").upper()
                if choice in ['Q', 'R', 'B', 'N']:
                    user_move.promotion_choice = choice
                
            if user_move:
                game_state.make_move(user_move)
            else:
                print("Invalid move. Try again.")
        else:
            print("CNN AI is thinking...")
            best_move = ai.get_best_move(game_state, valid_moves)
            print(f"CNN AI plays: {best_move.get_chess_notation()}")
            game_state.make_move(best_move)

if __name__ == "__main__":
    main()