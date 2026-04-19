/// Board Evaluation Module
///
/// Ports the full evaluation from ai_player.py:
/// - Material counting
/// - Piece-Square Tables (middlegame + endgame king)
/// - Phase detection (queen-based + material threshold)
/// - Passed pawn bonus
/// - Rook on open file bonus
///
/// All values are in centipawns. Positive = White advantage.

use shakmaty::{Board, Color, Piece, Role, Square, Bitboard, File};

// ─── Piece Base Values (centipawns) ───────────────────────
const PAWN_VAL: i32   = 100;
const KNIGHT_VAL: i32 = 320;
const BISHOP_VAL: i32 = 330;
const ROOK_VAL: i32   = 500;
const QUEEN_VAL: i32  = 900;

fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn   => PAWN_VAL,
        Role::Knight => KNIGHT_VAL,
        Role::Bishop => BISHOP_VAL,
        Role::Rook   => ROOK_VAL,
        Role::Queen  => QUEEN_VAL,
        Role::King   => 0,
    }
}

// ─── Piece-Square Tables ──────────────────────────────────
// Indexed [rank_from_white_perspective][file], i.e. [0] = rank 8, [7] = rank 1.
// These are identical to the Python tables in ai_player.py.

#[rustfmt::skip]
const PAWN_PST: [[i32; 8]; 8] = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [ 5,  5, 10, 25, 25, 10,  5,  5],
    [ 0,  0,  0, 20, 20,  0,  0,  0],
    [ 5, -5,-10,  0,  0,-10, -5,  5],
    [ 5, 10, 10,-20,-20, 10, 10,  5],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
];

#[rustfmt::skip]
const KNIGHT_PST: [[i32; 8]; 8] = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
];

#[rustfmt::skip]
const BISHOP_PST: [[i32; 8]; 8] = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
];

#[rustfmt::skip]
const ROOK_PST: [[i32; 8]; 8] = [
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [ 0,  0,  0,  5,  5,  0,  0,  0],
];

#[rustfmt::skip]
const QUEEN_PST: [[i32; 8]; 8] = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20],
];

#[rustfmt::skip]
const KING_PST_MIDDLE: [[i32; 8]; 8] = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20],
];

#[rustfmt::skip]
const KING_PST_ENDGAME: [[i32; 8]; 8] = [
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50],
];

/// Looks up the PST bonus for a piece on a given square.
/// For Black pieces, the table is vertically mirrored.
fn pst_bonus(role: Role, sq: Square, color: Color, is_endgame: bool) -> i32 {
    let table = match role {
        Role::Pawn   => &PAWN_PST,
        Role::Knight => &KNIGHT_PST,
        Role::Bishop => &BISHOP_PST,
        Role::Rook   => &ROOK_PST,
        Role::Queen  => &QUEEN_PST,
        Role::King   => if is_endgame { &KING_PST_ENDGAME } else { &KING_PST_MIDDLE },
    };

    // shakmaty squares: A1=0, H8=63. rank 0 = rank 1 (bottom).
    let rank = sq.rank() as usize; // 0 = rank 1, 7 = rank 8
    let file = sq.file() as usize; // 0 = a-file, 7 = h-file

    // PST is from White's perspective: index [0] = rank 8, [7] = rank 1.
    let row = match color {
        Color::White => 7 - rank, // White: rank 1 → row 7, rank 8 → row 0
        Color::Black => rank,     // Black: mirror vertically
    };

    table[row][file]
}

// ─── Passed Pawn Detection ────────────────────────────────
fn is_passed_pawn(board: &Board, sq: Square, color: Color) -> bool {
    let file = sq.file();
    let rank = sq.rank() as i32;
    let enemy_pawns = board.by_piece(Piece { color: !color, role: Role::Pawn });

    // Check the pawn's file and adjacent files for enemy pawns ahead
    let files_to_check: Vec<File> = {
        let f = file as i32;
        let mut files = vec![file];
        if f > 0 { files.push(File::new((f - 1) as u32)); }
        if f < 7 { files.push(File::new((f + 1) as u32)); }
        files
    };

    for &check_file in &files_to_check {
        let file_bb = Bitboard::from(check_file);
        let blockers = enemy_pawns & file_bb;

        for blocker_sq in blockers {
            let blocker_rank = blocker_sq.rank() as i32;
            match color {
                Color::White => {
                    if blocker_rank > rank { return false; } // Enemy pawn ahead
                }
                Color::Black => {
                    if blocker_rank < rank { return false; } // Enemy pawn ahead
                }
            }
        }
    }

    true
}

// ─── Rook on Open File ───────────────────────────────────

/// Checks if a rook's file has no friendly pawns.
fn is_open_file_for_rook(board: &Board, sq: Square, color: Color) -> bool {
    let file = sq.file();
    let friendly_pawns = board.by_piece(Piece { color, role: Role::Pawn });
    let file_bb = Bitboard::from(file);
    (friendly_pawns & file_bb).is_empty()
}

// ─── Main Evaluation ─────────────────────────────────────

/// Evaluates the board position. Returns score in centipawns.
/// Positive = White advantage. Negative = Black advantage.
pub fn evaluate(board: &Board) -> i32 {
    // Phase detection
    let white_queens = board.by_piece(Piece { color: Color::White, role: Role::Queen });
    let black_queens = board.by_piece(Piece { color: Color::Black, role: Role::Queen });
    let queens_dead = white_queens.is_empty() && black_queens.is_empty();

    let mut non_pawn_material = 0i32;
    for color in [Color::White, Color::Black] {
        for role in [Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
            let count = board.by_piece(Piece { color, role }).count() as i32;
            non_pawn_material += count * piece_value(role);
        }
    }

    let is_endgame = queens_dead || non_pawn_material < 1200;

    // Accumulate score
    let mut score: i32 = 0;

    for sq in board.occupied() {
        let piece = board.piece_at(sq).unwrap();
        let base = piece_value(piece.role);
        let pst = pst_bonus(piece.role, sq, piece.color, is_endgame);
        let mut val = base + pst;

        // Passed pawn bonus
        if piece.role == Role::Pawn && is_passed_pawn(board, sq, piece.color) {
            val += 50;
        }

        // Rook on open file bonus
        if piece.role == Role::Rook && is_open_file_for_rook(board, sq, piece.color) {
            val += 25;
        }

        match piece.color {
            Color::White => score += val,
            Color::Black => score -= val,
        }
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::{Chess, CastlingMode, Position};

    fn eval_fen(fen: &str) -> i32 {
        let pos: Chess = fen.parse::<Fen>().unwrap()
            .into_position(CastlingMode::Standard).unwrap();
        evaluate(pos.board())
    }

    #[test]
    fn starting_position_is_roughly_equal() {
        let score = eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(score.abs() < 50, "Start position should be ~equal, got {}", score);
    }

    #[test]
    fn white_up_a_queen_is_positive() {
        // White has extra queen
        let score = eval_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(score > 800, "White up a queen should be > 800, got {}", score);
    }

    #[test]
    fn black_up_a_queen_is_negative() {
        let score = eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1");
        assert!(score < -800, "Black up a queen should be < -800, got {}", score);
    }
}
