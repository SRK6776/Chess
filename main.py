from engine import GameState

def main():
    game_state = GameState()
    game_state.board_obj.draw_board()
    
    while True:
        moves = game_state.get_all_possible_moves()
        for move in moves:
            print(move.get_chess_notation())
        break

if __name__ == "__main__":
    main()