from board import Board

class Move:
    # Map ranks to rows, files to columns for human-readable output (e.g., "e2e4")
    ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_cols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_sq, end_sq, board):
        self.start_row = start_sq[0]
        self.start_col = start_sq[1]
        self.end_row = end_sq[0]
        self.end_col = end_sq[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        self.move_id = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col

    def get_chess_notation(self):
        return self.get_rank_file(self.start_row, self.start_col) + self.get_rank_file(self.end_row, self.end_col)

    def get_rank_file(self, r, c):
        return self.cols_to_files[c] + self.rows_to_ranks[r]

class GameState:
    def __init__(self):
        self.board_obj = Board()
        self.board = self.board_obj.board
        self.white_to_move = True
        self.move_log = []
        
        # This dictionary maps piece characters to their movement functions
        self.move_functions = {
            'P': self.get_pawn_moves, 'R': self.get_rook_moves, 
            'N': self.get_knight_moves, 'B': self.get_bishop_moves, 
            'Q': self.get_queen_moves, 'K': self.get_king_moves
        }

    def get_all_possible_moves(self):
        moves = []
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if piece != ".":
                    # Check if the piece matches the current turn color
                    is_white_piece = piece.isupper()
                    if (is_white_piece and self.white_to_move) or (not is_white_piece and not self.white_to_move):
                        piece_type = piece.upper()
                        self.move_functions[piece_type](r, c, moves)
        return moves

    def get_knight_moves(self, r, c, moves):
        # The 8 "L" shapes
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1), 
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        # Check if the piece at (r, c) is white or black
        ally_color = 'w' if self.board[r][c].isupper() else 'b'

        for dr, dc in knight_moves:
            end_row, end_col = r + dr, c + dc
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                target_piece = self.board[end_row][end_col]
                
                # Logical Check: Square must be empty OR have an enemy piece
                if target_piece == ".":
                    moves.append(Move((r, c), (end_row, end_col), self.board))
                else:
                    target_color = 'w' if target_piece.isupper() else 'b'
                    if target_color != ally_color:
                        moves.append(Move((r, c), (end_row, end_col), self.board))

    def get_rook_moves(self, r, c, moves):
        # Rooks move horizontally and vertically
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        ally_color = 'w' if self.board[r][c].isupper() else 'b'

        for dr, dc in directions:
            end_row, end_col = r + dr, c + dc
            
            # Keep moving in this direction until we hit a wall or another piece
            while 0 <= end_row < 8 and 0 <= end_col < 8:
                target_piece = self.board[end_row][end_col]
                
                if target_piece == ".":
                    # Empty square: Valid move, continue sliding
                    moves.append(Move((r, c), (end_row, end_col), self.board))
                    end_row += dr
                    end_col += dc
                else:
                    # Occupied square: Check if it's an enemy
                    target_color = 'w' if target_piece.isupper() else 'b'
                    if target_color != ally_color:
                        # Enemy piece: Valid capture, but stop sliding
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                    # Stop sliding regardless of whether it was an enemy or ally
                    break
    
    def get_bishop_moves(self, r, c, moves):
        # Bishops move diagonally
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        ally_color = 'w' if self.board[r][c].isupper() else 'b'

        for dr, dc in directions:
            end_row, end_col = r + dr, c + dc
            
            # Keep moving in this direction until we hit a wall or another piece
            while 0 <= end_row < 8 and 0 <= end_col < 8:
                target_piece = self.board[end_row][end_col]
                
                if target_piece == ".":
                    # Empty square: Valid move, continue sliding
                    moves.append(Move((r, c), (end_row, end_col), self.board))
                    end_row += dr
                    end_col += dc
                else:
                    # Occupied square: Check if it's an enemy
                    target_color = 'w' if target_piece.isupper() else 'b'
                    if target_color != ally_color:
                        # Enemy piece: Valid capture, but stop sliding
                        moves.append(Move((r, c), (end_row, end_col), self.board))
                    # Stop sliding regardless of whether it was an enemy or ally
                    break
    
    def get_queen_moves(self, r, c, moves):
        # Queen moves are just Rook moves + Bishop moves
        self.get_rook_moves(r, c, moves)
        self.get_bishop_moves(r, c, moves)

    def get_king_moves(self, r, c, moves):
        # The 8 surrounding squares
        king_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        ally_color = 'w' if self.board[r][c].isupper() else 'b'

        for dr, dc in king_moves:
            end_row, end_col = r + dr, c + dc
            if 0 <= end_row < 8 and 0 <= end_col < 8:
                target_piece = self.board[end_row][end_col]
                if target_piece == ".":
                    moves.append(Move((r, c), (end_row, end_col), self.board))
                else:
                    target_color = 'w' if target_piece.isupper() else 'b'
                    if target_color != ally_color:
                        moves.append(Move((r, c), (end_row, end_col), self.board))

    def get_pawn_moves(self, r, c, moves):
        if self.white_to_move:  # White Pawn Moves
            # 1. Advance one square
            if self.board[r-1][c] == ".":
                moves.append(Move((r, c), (r-1, c), self.board))
                # 2. Advance two squares from starting rank (6)
                if r == 6 and self.board[r-2][c] == ".":
                    moves.append(Move((r, c), (r-2, c), self.board))
            # 3. Captures
            if c-1 >= 0:  # Capture left
                if self.board[r-1][c-1].islower():
                    moves.append(Move((r, c), (r-1, c-1), self.board))
            if c+1 <= 7:  # Capture right
                if self.board[r-1][c+1].islower():
                    moves.append(Move((r, c), (r-1, c+1), self.board))

        else:  # Black Pawn Moves
            # 1. Advance one square
            if self.board[r+1][c] == ".":
                moves.append(Move((r, c), (r+1, c), self.board))
                # 2. Advance two squares from starting rank (1)
                if r == 1 and self.board[r+2][c] == ".":
                    moves.append(Move((r, c), (r+2, c), self.board))
            # 3. Captures
            if c-1 >= 0:
                if self.board[r+1][c-1].isupper():
                    moves.append(Move((r, c), (r+1, c-1), self.board))
            if c+1 <= 7:
                if self.board[r+1][c+1].isupper():
                    moves.append(Move((r, c), (r+1, c+1), self.board))