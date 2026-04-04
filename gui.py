import pygame
import sys
import copy
import threading
from engine import GameState, Move
from ai_player import AIPlayer
import os

# ─── Constants ─────────────────────────────────────────────
BOARD_SIZE = 640
SQ_SIZE = BOARD_SIZE // 8
PANEL_WIDTH = 280
WINDOW_WIDTH = BOARD_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE
FPS = 60

# ─── Color Palette ─────────────────────────────────────────
# Board
LIGHT_SQ = (234, 221, 202)
DARK_SQ = (168, 121, 85)
HIGHLIGHT_COLOR = (186, 202, 68, 180)     # Yellow-green for selected square
VALID_MOVE_COLOR = (100, 180, 80, 140)    # Green dots for legal moves
LAST_MOVE_COLOR = (205, 210, 106, 150)    # Subtle tan for last move
CHECK_COLOR = (220, 50, 50, 180)          # Red highlight for king in check

# Panel
PANEL_BG = (32, 32, 36)
PANEL_TEXT = (210, 210, 215)
PANEL_ACCENT = (100, 180, 100)
PANEL_DIVIDER = (60, 60, 65)
STATUS_WIN = (100, 200, 100)
STATUS_DRAW = (200, 200, 100)

# ─── Piece Images ──────────────────────────────────────────
PIECE_IMAGES = {}

def load_piece_images():
    """Loads all 12 chess piece PNGs from images/ and scales them to SQ_SIZE."""
    piece_map = {
        'K': 'wK', 'Q': 'wQ', 'R': 'wR', 'B': 'wB', 'N': 'wN', 'P': 'wP',
        'k': 'bK', 'q': 'bQ', 'r': 'bR', 'b': 'bB', 'n': 'bN', 'p': 'bP',
    }
    for piece_char, filename in piece_map.items():
        path = os.path.join("images", f"{filename}.png")
        img = pygame.image.load(path).convert_alpha()
        PIECE_IMAGES[piece_char] = pygame.transform.smoothscale(img, (SQ_SIZE, SQ_SIZE))

def draw_piece_surface(piece_char, size):
    """Returns the cached piece image (or empty surface if not found)."""
    if piece_char in PIECE_IMAGES:
        if size == SQ_SIZE:
            return PIECE_IMAGES[piece_char]
        else:
            return pygame.transform.smoothscale(PIECE_IMAGES[piece_char], (size, size))
    return pygame.Surface((size, size), pygame.SRCALPHA)

# ─── Board Drawing ─────────────────────────────────────────
def draw_board(screen, game_state, board_2d, selected_sq, valid_moves, last_move, dragging_piece, drag_pos):
    """Draws the chessboard, pieces, and all visual overlays.
    board_2d: the 8x8 list to render (may be a frozen snapshot during AI thinking)."""
    
    for r in range(8):
        for c in range(8):
            # Base square color
            color = LIGHT_SQ if (r + c) % 2 == 0 else DARK_SQ
            rect = pygame.Rect(c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE)
            pygame.draw.rect(screen, color, rect)
            
            # Highlight last move
            if last_move:
                if (r, c) == (last_move.start_row, last_move.start_col) or \
                   (r, c) == (last_move.end_row, last_move.end_col):
                    overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    overlay.fill(LAST_MOVE_COLOR)
                    screen.blit(overlay, rect)
    
    # Highlight king if in check
    if game_state.is_in_check():
        if game_state.white_to_move:
            kr, kc = game_state.white_king_location
        else:
            kr, kc = game_state.black_king_location
        overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        overlay.fill(CHECK_COLOR)
        screen.blit(overlay, (kc * SQ_SIZE, kr * SQ_SIZE))
    
    # Highlight selected square
    if selected_sq:
        overlay = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
        overlay.fill(HIGHLIGHT_COLOR)
        screen.blit(overlay, (selected_sq[1] * SQ_SIZE, selected_sq[0] * SQ_SIZE))
        
        # Draw valid move indicators
        for move in valid_moves:
            if (move.start_row, move.start_col) == selected_sq:
                center_x = move.end_col * SQ_SIZE + SQ_SIZE // 2
                center_y = move.end_row * SQ_SIZE + SQ_SIZE // 2
                
                if move.piece_captured != ".":
                    # Draw ring for captures
                    ring_surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(ring_surf, VALID_MOVE_COLOR, (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 2 - 4, 4)
                    screen.blit(ring_surf, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))
                else:
                    # Draw dot for quiet moves
                    dot_surf = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
                    pygame.draw.circle(dot_surf, VALID_MOVE_COLOR, (SQ_SIZE // 2, SQ_SIZE // 2), SQ_SIZE // 6)
                    screen.blit(dot_surf, (move.end_col * SQ_SIZE, move.end_row * SQ_SIZE))

    # Draw pieces from the provided board_2d (may be a snapshot)
    for r in range(8):
        for c in range(8):
            piece = board_2d[r][c]
            if piece != ".":
                # Skip drawing piece being dragged at its original position
                if dragging_piece and (r, c) == selected_sq:
                    continue
                surf = draw_piece_surface(piece, SQ_SIZE)
                screen.blit(surf, (c * SQ_SIZE, r * SQ_SIZE))

    # Draw dragged piece following cursor
    if dragging_piece and drag_pos:
        surf = draw_piece_surface(dragging_piece, SQ_SIZE)
        screen.blit(surf, (drag_pos[0] - SQ_SIZE // 2, drag_pos[1] - SQ_SIZE // 2))

    # Rank/File labels
    label_font = pygame.font.SysFont("arial", 12, bold=True)
    for i in range(8):
        # Files (a-h) along bottom
        label = label_font.render(chr(ord('a') + i), True, DARK_SQ if i % 2 == 0 else LIGHT_SQ)
        screen.blit(label, (i * SQ_SIZE + SQ_SIZE - 14, BOARD_SIZE - 16))
        # Ranks (8-1) along left
        label = label_font.render(str(8 - i), True, LIGHT_SQ if i % 2 == 0 else DARK_SQ)
        screen.blit(label, (3, i * SQ_SIZE + 3))


# ─── Side Panel ────────────────────────────────────────────
def draw_panel(screen, game_state, ai_eval, ai_thinking):
    panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    
    # Divider line
    pygame.draw.line(screen, PANEL_DIVIDER, (BOARD_SIZE, 0), (BOARD_SIZE, WINDOW_HEIGHT), 2)
    
    x = BOARD_SIZE + 20
    y = 20
    
    # Title
    title_font = pygame.font.SysFont("arial", 22, bold=True)
    title = title_font.render("Chess Engine", True, PANEL_TEXT)
    screen.blit(title, (x, y))
    y += 36
    
    subtitle_font = pygame.font.SysFont("arial", 13)
    sub = subtitle_font.render("CNN + Minimax Hybrid AI", True, PANEL_ACCENT)
    screen.blit(sub, (x, y))
    y += 40
    
    # Divider
    pygame.draw.line(screen, PANEL_DIVIDER, (x, y), (BOARD_SIZE + PANEL_WIDTH - 20, y), 1)
    y += 20
    
    # Info section
    info_font = pygame.font.SysFont("arial", 15)
    
    # Turn indicator
    turn_text = "White to move" if game_state.white_to_move else "Black to move"
    turn_color = (255, 255, 255) if game_state.white_to_move else (180, 180, 180)
    screen.blit(info_font.render(turn_text, True, turn_color), (x, y))
    y += 26
    
    # Move count
    move_num = len(game_state.move_log) // 2 + 1
    screen.blit(info_font.render(f"Move: {move_num}", True, PANEL_TEXT), (x, y))
    y += 26
    
    # AI evaluation
    if ai_thinking:
        screen.blit(info_font.render("AI is thinking...", True, (255, 200, 80)), (x, y))
    elif ai_eval is not None:
        eval_str = f"AI Eval: {ai_eval:+.2f}"
        eval_color = STATUS_WIN if ai_eval < 0 else (200, 100, 100) if ai_eval > 0 else PANEL_TEXT
        screen.blit(info_font.render(eval_str, True, eval_color), (x, y))
    y += 40
    
    # Divider
    pygame.draw.line(screen, PANEL_DIVIDER, (x, y), (BOARD_SIZE + PANEL_WIDTH - 20, y), 1)
    y += 20
    
    # Game status
    if game_state.checkmate:
        winner = "Black" if game_state.white_to_move else "White"
        status_font = pygame.font.SysFont("arial", 18, bold=True)
        screen.blit(status_font.render(f"Checkmate!", True, STATUS_WIN), (x, y))
        y += 26
        screen.blit(info_font.render(f"{winner} wins!", True, STATUS_WIN), (x, y))
    elif game_state.stalemate:
        status_font = pygame.font.SysFont("arial", 18, bold=True)
        screen.blit(status_font.render("Stalemate!", True, STATUS_DRAW), (x, y))
        y += 26
        screen.blit(info_font.render("Draw", True, STATUS_DRAW), (x, y))
    else:
        in_check = game_state.is_in_check()
        if in_check:
            screen.blit(info_font.render("Check!", True, (220, 80, 80)), (x, y))
            y += 26
    
    # Move log (last 12 moves)
    y = WINDOW_HEIGHT - 260
    pygame.draw.line(screen, PANEL_DIVIDER, (x, y), (BOARD_SIZE + PANEL_WIDTH - 20, y), 1)
    y += 10
    
    log_title = pygame.font.SysFont("arial", 14, bold=True)
    screen.blit(log_title.render("Move History", True, PANEL_TEXT), (x, y))
    y += 24
    
    log_font = pygame.font.SysFont("consolas,courier", 13)
    moves = game_state.move_log
    start_idx = max(0, len(moves) - 12)
    
    for i in range(start_idx, len(moves), 2):
        move_num = i // 2 + 1
        white_move = moves[i].get_chess_notation() if i < len(moves) else ""
        black_move = moves[i + 1].get_chess_notation() if i + 1 < len(moves) else ""
        line = f"{move_num:>3}. {white_move:<6} {black_move:<6}"
        screen.blit(log_font.render(line, True, (160, 160, 165)), (x, y))
        y += 18
        
    # Controls hint
    hint_font = pygame.font.SysFont("arial", 11)
    hint_y = WINDOW_HEIGHT - 25
    screen.blit(hint_font.render("Z = Undo  |  R = Reset  |  ESC = Quit", True, (90, 90, 95)), (x, hint_y))


# ─── Promotion Dialog ─────────────────────────────────────
def show_promotion_dialog(screen, is_white):
    """Shows a simple promotion picker and returns the chosen piece."""
    pieces = ['Q', 'R', 'B', 'N']
    piece_chars = [p if is_white else p.lower() for p in pieces]
    
    # Overlay background
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))
    
    # Dialog box
    box_w, box_h = 320, 120
    box_x = (BOARD_SIZE - box_w) // 2
    box_y = (BOARD_SIZE - box_h) // 2
    pygame.draw.rect(screen, PANEL_BG, (box_x, box_y, box_w, box_h), border_radius=10)
    pygame.draw.rect(screen, PANEL_ACCENT, (box_x, box_y, box_w, box_h), 2, border_radius=10)
    
    # Title
    font = pygame.font.SysFont("arial", 16, bold=True)
    title = font.render("Promote Pawn To:", True, PANEL_TEXT)
    screen.blit(title, (box_x + (box_w - title.get_width()) // 2, box_y + 10))
    
    # Piece buttons
    btn_size = 60
    total_w = btn_size * 4 + 12 * 3
    start_x = box_x + (box_w - total_w) // 2
    btn_y = box_y + 45
    
    btn_rects = []
    for i, pc in enumerate(piece_chars):
        bx = start_x + i * (btn_size + 12)
        rect = pygame.Rect(bx, btn_y, btn_size, btn_size)
        pygame.draw.rect(screen, (60, 60, 65), rect, border_radius=6)
        pygame.draw.rect(screen, PANEL_ACCENT, rect, 2, border_radius=6)
        
        surf = draw_piece_surface(pc, btn_size)
        screen.blit(surf, (bx, btn_y))
        btn_rects.append((rect, pieces[i]))
    
    pygame.display.flip()
    
    # Wait for user click
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for rect, piece in btn_rects:
                    if rect.collidepoint(event.pos):
                        return piece
    return 'Q'


# ─── Main Game Loop ────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Chess — CNN AI Engine")
    load_piece_images()
    clock = pygame.time.Clock()

    game_state = GameState()
    
    # Load AI
    if os.path.exists("chess_model.pth"):
        ai = AIPlayer("chess_model.pth")
    else:
        print("Model not trained yet! Run train_model.py first.")
        return

    player_white = True
    valid_moves = game_state.get_valid_moves()
    selected_sq = None
    last_move = None
    ai_eval = None
    ai_thinking = False
    ai_move_ready = None
    dragging_piece = None
    drag_pos = None
    game_over = False
    board_snapshot = None  # Frozen board to display while AI is thinking

    while True:
        is_human_turn = (game_state.white_to_move and player_white) or \
                        (not game_state.white_to_move and not player_white)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_z and not ai_thinking:
                    # Undo
                    if len(game_state.move_log) >= 2:
                        game_state.undo_move()
                        game_state.undo_move()
                    elif len(game_state.move_log) == 1:
                        game_state.undo_move()
                    valid_moves = game_state.get_valid_moves()
                    selected_sq = None
                    last_move = game_state.move_log[-1] if game_state.move_log else None
                    game_over = False
                    ai_eval = None
                    
                if event.key == pygame.K_r:
                    # Reset
                    game_state = GameState()
                    valid_moves = game_state.get_valid_moves()
                    selected_sq = None
                    last_move = None
                    ai_eval = None
                    ai_thinking = False
                    ai_move_ready = None
                    game_over = False
            
            if event.type == pygame.MOUSEBUTTONDOWN and is_human_turn and not game_over and not ai_thinking:
                mx, my = event.pos
                if mx < BOARD_SIZE:
                    c = mx // SQ_SIZE
                    r = my // SQ_SIZE
                    clicked_piece = game_state.board[r][c]
                    
                    if selected_sq is None:
                        # First click: select a piece
                        if clicked_piece != ".":
                            is_own_piece = (clicked_piece.isupper() and game_state.white_to_move) or \
                                           (clicked_piece.islower() and not game_state.white_to_move)
                            if is_own_piece:
                                selected_sq = (r, c)
                                dragging_piece = clicked_piece
                                drag_pos = (mx, my)
                    else:
                        # Clicked a different own piece — reselect
                        if clicked_piece != ".":
                            is_own_piece = (clicked_piece.isupper() and game_state.white_to_move) or \
                                           (clicked_piece.islower() and not game_state.white_to_move)
                            if is_own_piece and (r, c) != selected_sq:
                                selected_sq = (r, c)
                                dragging_piece = clicked_piece
                                drag_pos = (mx, my)
                                continue
                        
                        # Second click: attempt move
                        target_sq = (r, c)
                        user_move = None
                        for move in valid_moves:
                            if (move.start_row, move.start_col) == selected_sq and \
                               (move.end_row, move.end_col) == target_sq:
                                user_move = move
                                break
                        
                        if user_move:
                            if user_move.is_promotion_move:
                                choice = show_promotion_dialog(screen, game_state.white_to_move)
                                user_move.promotion_choice = choice
                            game_state.make_move(user_move)
                            last_move = user_move
                            valid_moves = game_state.get_valid_moves()
                            if game_state.checkmate or game_state.stalemate:
                                game_over = True
                        selected_sq = None
                        dragging_piece = None
                        drag_pos = None
            
            if event.type == pygame.MOUSEMOTION and dragging_piece:
                drag_pos = event.pos
            
            if event.type == pygame.MOUSEBUTTONUP and dragging_piece and is_human_turn and not game_over:
                mx, my = event.pos
                if mx < BOARD_SIZE:
                    r = my // SQ_SIZE
                    c = mx // SQ_SIZE
                    target_sq = (r, c)
                    
                    if target_sq != selected_sq:
                        user_move = None
                        for move in valid_moves:
                            if (move.start_row, move.start_col) == selected_sq and \
                               (move.end_row, move.end_col) == target_sq:
                                user_move = move
                                break
                        
                        if user_move:
                            if user_move.is_promotion_move:
                                choice = show_promotion_dialog(screen, game_state.white_to_move)
                                user_move.promotion_choice = choice
                            game_state.make_move(user_move)
                            last_move = user_move
                            valid_moves = game_state.get_valid_moves()
                            if game_state.checkmate or game_state.stalemate:
                                game_over = True
                            selected_sq = None
                            dragging_piece = None
                            drag_pos = None
                        else:
                            # Invalid drop — keep piece selected (click-click fallback)
                            dragging_piece = None
                            drag_pos = None
                else:
                    dragging_piece = None
                    drag_pos = None

        # ─── AI Turn ───────────────────────────────────────
        if not is_human_turn and not game_over and not ai_thinking and ai_move_ready is None:
            ai_thinking = True
            board_snapshot = copy.deepcopy(game_state.board)  # Freeze what the player sees
            
            def ai_thread_func():
                nonlocal ai_move_ready, ai_eval
                best = ai.get_best_move(game_state, valid_moves)
                ai_move_ready = best
            
            thread = threading.Thread(target=ai_thread_func, daemon=True)
            thread.start()
        
        if ai_move_ready is not None:
            game_state.make_move(ai_move_ready)
            last_move = ai_move_ready
            valid_moves = game_state.get_valid_moves()
            if game_state.checkmate or game_state.stalemate:
                game_over = True
            ai_move_ready = None
            ai_thinking = False
            board_snapshot = None  # Unfreeze — show live board again

        # ─── Draw ──────────────────────────────────────────
        display_board = board_snapshot if board_snapshot is not None else game_state.board
        draw_board(screen, game_state, display_board, selected_sq, valid_moves, last_move, dragging_piece, drag_pos)
        draw_panel(screen, game_state, ai_eval, ai_thinking)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
