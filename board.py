class Board:
    def __init__(self):
        # Pieces: 'P'=Pawn, 'N'=Knight, 'B'=Bishop, 'R'=Rook, 'Q'=Queen, 'K'=King
        # Case: Upper = White, Lower = Black, '.' = Empty
        self.board = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", "p", "p", "p", "p"],
            [".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", "."],
            [".", ".", ".", ".", ".", ".", ".", "."],
            ["P", "P", "P", "P", "P", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"]
        ]

    def draw_board(self):
        for row in self.board:
            print(" ".join(row))

    def load_fen(self, fen):
        pieces, turn, castling, en_passant, half_move, full_move = fen.split()
        
        # 1. Update Board
        rows = pieces.split('/')
        for r in range(8):
            col = 0
            for char in rows[r]:
                if char.isdigit():
                    for _ in range(int(char)):
                        self.board[r][col] = "."
                        col += 1
                else:
                    self.board[r][col] = char
                    col += 1
        
        # 2. Update Metadata
        self.white_to_move = (turn == 'w')
    
    def get_piece(self, square):
        return self.board[square[0]][square[1]]

    def is_empty(self, square):
        return self.get_piece(square) == "."
