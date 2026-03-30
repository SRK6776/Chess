from board import Board

class Move:
    # Map ranks to rows, files to columns for human-readable output (e.g., "e2e4")
    ranks_to_rows = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}
    rows_to_ranks = {v: k for k, v in ranks_to_rows.items()}
    files_to_cols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    cols_to_files = {v: k for k, v in files_to_cols.items()}

    def __init__(self, start_sq, end_sq, board, en_passant_move=False):
        self.start_row = start_sq[0]
        self.start_col = start_sq[1]
        self.end_row = end_sq[0]
        self.end_col = end_sq[1]
        self.piece_moved = board[self.start_row][self.start_col]
        self.piece_captured = board[self.end_row][self.end_col]
        self.move_id = self.start_row * 1000 + self.start_col * 100 + self.end_row * 10 + self.end_col
        self.is_castle_move = False

        # En passant: destination is empty so manually record the captured pawn
        self.is_en_passant_move = en_passant_move
        if self.is_en_passant_move:
            self.piece_captured = 'p' if self.piece_moved == 'P' else 'P'

        # Promotion: pawn reaches the back rank
        self.is_promotion_move = (
            (self.piece_moved == 'P' and self.end_row == 0) or
            (self.piece_moved == 'p' and self.end_row == 7)
        )

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
        self.white_king_location = (7, 4)
        self.black_king_location = (0, 4)
        self.checkmate = False
        self.stalemate = False
        self.en_passant_square = ()  # (row, col)
        
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
            # 1. Advance one square (guard r > 0 to avoid index wrap)
            if r > 0 and self.board[r-1][c] == ".":
                moves.append(Move((r, c), (r-1, c), self.board))
                # 2. Advance two squares from starting rank (row 6)
                if r == 6 and self.board[r-2][c] == ".":
                    moves.append(Move((r, c), (r-2, c), self.board))
            # 3. Diagonal captures
            if r > 0:
                if c-1 >= 0 and self.board[r-1][c-1].islower():  # Capture left
                    moves.append(Move((r, c), (r-1, c-1), self.board))
                if c+1 <= 7 and self.board[r-1][c+1].islower():  # Capture right
                    moves.append(Move((r, c), (r-1, c+1), self.board))
            # 4. En passant captures
            if r == 3:  # White can only en passant from rank 5 (row 3)
                if c-1 >= 0 and self.en_passant_square == (r-1, c-1):
                    moves.append(Move((r, c), (r-1, c-1), self.board, en_passant_move=True))
                if c+1 <= 7 and self.en_passant_square == (r-1, c+1):
                    moves.append(Move((r, c), (r-1, c+1), self.board, en_passant_move=True))

        else:  # Black Pawn Moves
            # 1. Advance one square
            if r < 7 and self.board[r+1][c] == ".":
                moves.append(Move((r, c), (r+1, c), self.board))
                # 2. Advance two squares from starting rank (row 1)
                if r == 1 and self.board[r+2][c] == ".":
                    moves.append(Move((r, c), (r+2, c), self.board))
            # 3. Diagonal captures
            if r < 7:
                if c-1 >= 0 and self.board[r+1][c-1].isupper():  # Capture left
                    moves.append(Move((r, c), (r+1, c-1), self.board))
                if c+1 <= 7 and self.board[r+1][c+1].isupper():  # Capture right
                    moves.append(Move((r, c), (r+1, c+1), self.board))
            # 4. En passant captures
            if r == 4:  # Black can only en passant from rank 4 (row 4)
                if c-1 >= 0 and self.en_passant_square == (r+1, c-1):
                    moves.append(Move((r, c), (r+1, c-1), self.board, en_passant_move=True))
                if c+1 <= 7 and self.en_passant_square == (r+1, c+1):
                    moves.append(Move((r, c), (r+1, c+1), self.board, en_passant_move=True))


    def make_move(self, move):
        self.board[move.start_row][move.start_col] = "."
        self.board[move.end_row][move.end_col] = move.piece_moved
        self.move_log.append(move)

        # Track king positions
        if move.piece_moved == 'K':
            self.white_king_location = (move.end_row, move.end_col)
        elif move.piece_moved == 'k':
            self.black_king_location = (move.end_row, move.end_col)

        # En passant: remove the captured pawn (it sits on start_row, end_col — not end_row)
        if move.is_en_passant_move:
            self.board[move.start_row][move.end_col] = "."

        # Update en passant square (only valid immediately after a double pawn push)
        if move.piece_moved == 'P' and move.start_row - move.end_row == 2:
            self.en_passant_square = ((move.start_row + move.end_row) // 2, move.start_col)
        elif move.piece_moved == 'p' and move.end_row - move.start_row == 2:
            self.en_passant_square = ((move.start_row + move.end_row) // 2, move.start_col)
        else:
            self.en_passant_square = ()  # Reset after any non-double-push move

        # Pawn promotion: read whose turn it is BEFORE flipping
        if move.is_promotion_move:
            is_white = self.white_to_move
            choice = input("Promote to (Q, R, B, N): ").upper()
            piece_map = {'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N'}
            promoted = piece_map.get(choice, 'Q')
            self.board[move.end_row][move.end_col] = promoted if is_white else promoted.lower()

        self.white_to_move = not self.white_to_move

    def undo_move(self):
        if len(self.move_log) == 0:
            return
        move = self.move_log.pop()
        self.board[move.start_row][move.start_col] = move.piece_moved
        self.board[move.end_row][move.end_col] = move.piece_captured
        self.white_to_move = not self.white_to_move

        # Restore king positions
        if move.piece_moved == 'K':
            self.white_king_location = (move.start_row, move.start_col)
        elif move.piece_moved == 'k':
            self.black_king_location = (move.start_row, move.start_col)

        # Restore en passant captured pawn (was removed from start_row, end_col)
        if move.is_en_passant_move:
            self.board[move.end_row][move.end_col] = "."        # destination was empty
            self.board[move.start_row][move.end_col] = move.piece_captured  # restore pawn

        # Restore en passant square from the previous move
        if len(self.move_log) > 0:
            prev = self.move_log[-1]
            if prev.piece_moved == 'P' and prev.start_row - prev.end_row == 2:
                self.en_passant_square = ((prev.start_row + prev.end_row) // 2, prev.start_col)
            elif prev.piece_moved == 'p' and prev.end_row - prev.start_row == 2:
                self.en_passant_square = ((prev.start_row + prev.end_row) // 2, prev.start_col)
            else:
                self.en_passant_square = ()
        else:
            self.en_passant_square = ()

    def is_in_check(self):
        if self.white_to_move:
            return self.is_square_attacked(self.white_king_location[0], self.white_king_location[1])
        else:
            return self.is_square_attacked(self.black_king_location[0], self.black_king_location[1])

    def is_square_attacked(self, r, c):
        self.white_to_move = not self.white_to_move
        opp_moves = self.get_all_possible_moves()
        self.white_to_move = not self.white_to_move
        for move in opp_moves:
            if move.end_row == r and move.end_col == c:
                return True
        return False

    def get_valid_moves(self):
        # 1. Generate all possible moves based on piece rules
        moves = self.get_all_possible_moves()
        
        # 2. We must iterate backwards because we are removing items from the list
        for i in range(len(moves) - 1, -1, -1):
            # 3. "Test drive" the move
            self.make_move(moves[i])
            
            # 4. After we move, it's the opponent's turn. 
            # Check if OUR king is now under attack.
            self.white_to_move = not self.white_to_move
            if self.is_in_check():
                moves.remove(moves[i]) # Illegal move!
            self.white_to_move = not self.white_to_move
            
            # 5. Undo the move to restore the board
            self.undo_move()

        if len(moves) == 0:
            if self.is_in_check():
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.checkmate = False  # Reset if legal moves exist (e.g. after undo)
            self.stalemate = False

        return moves
