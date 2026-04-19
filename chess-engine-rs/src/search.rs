/// Search Module — Minimax + Alpha-Beta + Quiescence
///
/// Ports the search logic from ai_player.py:
/// - fast_minimax with alpha-beta pruning
/// - Quiescence search (captures only, solves horizon effect)
/// - MVV-LVA move ordering for pruning efficiency
/// - Checkmate/stalemate detection at every node

use shakmaty::{Chess, Color, Move, Position, Role};
use crate::eval;

const MATE_SCORE: i32 = 999_999;

// ─── MVV-LVA Move Ordering ───────────────────────────────

/// Scores a move for ordering. Higher = search first.
/// Captures are scored by MVV-LVA (Most Valuable Victim - Least Valuable Attacker).
fn move_order_score(_pos: &Chess, mv: &Move) -> i32 {
    if mv.is_capture() {
        let victim_val = match mv.capture() {
            Some(role) => role_value(role),
            None => 0, // En passant is handled by shakmaty
        };
        let attacker_val = role_value(mv.role());
        // MVV-LVA: prioritize capturing expensive pieces with cheap ones
        victim_val * 10 - attacker_val + 10_000 // +10000 ensures captures > quiet moves
    } else if mv.is_promotion() {
        9_000 // Promotions are almost always good
    } else {
        0
    }
}

fn role_value(role: Role) -> i32 {
    match role {
        Role::Pawn   => 100,
        Role::Knight => 320,
        Role::Bishop => 330,
        Role::Rook   => 500,
        Role::Queen  => 900,
        Role::King   => 0,
    }
}

// ─── Quiescence Search ───────────────────────────────────

/// Continues searching captures until the position is "quiet".
/// This prevents the horizon effect where the engine stops evaluating
/// in the middle of a capture sequence.
fn quiescence(pos: &Chess, mut alpha: i32, beta: i32, is_maximizing: bool) -> i32 {
    let legals = pos.legal_moves();

    // Terminal node check
    if legals.is_empty() {
        if pos.is_check() {
            return if is_maximizing { -MATE_SCORE } else { MATE_SCORE };
        }
        return 0; // Stalemate
    }

    let stand_pat = eval::evaluate(pos.board());

    if is_maximizing {
        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }

        // Only search captures
        let mut captures: Vec<&Move> = legals.iter().filter(|m| m.is_capture()).collect();
        captures.sort_by(|a, b| move_order_score(pos, b).cmp(&move_order_score(pos, a)));

        for mv in captures {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = quiescence(&child, alpha, beta, false);

            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }
        alpha
    } else {
        let mut beta = beta;
        if stand_pat <= alpha { return alpha; }
        if stand_pat < beta { beta = stand_pat; }

        let mut captures: Vec<&Move> = legals.iter().filter(|m| m.is_capture()).collect();
        captures.sort_by(|a, b| move_order_score(pos, b).cmp(&move_order_score(pos, a)));

        for mv in captures {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = quiescence(&child, alpha, beta, true);

            if score <= alpha { return alpha; }
            if score < beta { beta = score; }
        }
        beta
    }
}

// ─── Minimax with Alpha-Beta Pruning ─────────────────────

/// Core search function. Identical logic to fast_minimax in ai_player.py
/// but runs ~100x faster due to bitboard move generation and compiled code.
fn minimax(pos: &Chess, depth: u32, mut alpha: i32, mut beta: i32, is_maximizing: bool) -> i32 {
    if depth == 0 {
        return quiescence(pos, alpha, beta, is_maximizing);
    }

    let legals = pos.legal_moves();

    // Terminal node: checkmate or stalemate
    if legals.is_empty() {
        if pos.is_check() {
            return if is_maximizing { -MATE_SCORE } else { MATE_SCORE };
        }
        return 0; // Stalemate
    }

    // Sort moves for better pruning
    let mut moves: Vec<&Move> = legals.iter().collect();
    moves.sort_by(|a, b| move_order_score(pos, b).cmp(&move_order_score(pos, a)));

    if is_maximizing {
        let mut max_eval = i32::MIN;
        for mv in moves {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = minimax(&child, depth - 1, alpha, beta, false);
            max_eval = max_eval.max(score);
            alpha = alpha.max(score);
            if beta <= alpha { break; } // Beta cutoff
        }
        max_eval
    } else {
        let mut min_eval = i32::MAX;
        for mv in moves {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = minimax(&child, depth - 1, alpha, beta, true);
            min_eval = min_eval.min(score);
            beta = beta.min(score);
            if beta <= alpha { break; } // Alpha cutoff
        }
        min_eval
    }
}

// ─── Public Interface ────────────────────────────────────

/// Result of a search: the best move and its evaluation score.
pub struct SearchResult {
    pub best_move: Move,
    pub score: i32,
}

/// Runs pure tactical search (no CNN). Used as fallback if model is unavailable.
pub fn find_best_move(pos: &Chess, depth: u32) -> Option<SearchResult> {
    let legals = pos.legal_moves();
    if legals.is_empty() {
        return None;
    }

    let is_maximizing = pos.turn() == Color::White;

    let mut moves: Vec<&Move> = legals.iter().collect();
    moves.sort_by(|a, b| move_order_score(pos, b).cmp(&move_order_score(pos, a)));

    let mut best_move = moves[0].clone();
    let mut best_score = if is_maximizing { i32::MIN } else { i32::MAX };

    for mv in moves {
        let mut child = pos.clone();
        child.play_unchecked(mv);

        let score = minimax(&child, depth - 1, i32::MIN, i32::MAX, !is_maximizing);

        if is_maximizing {
            if score > best_score {
                best_score = score;
                best_move = mv.clone();
            }
        } else {
            if score < best_score {
                best_score = score;
                best_move = mv.clone();
            }
        }
    }

    Some(SearchResult {
        best_move,
        score: best_score,
    })
}

/// Runs hybrid search: tactical minimax blended with CNN style probabilities.
/// This is the direct port of get_best_move() in ai_player.py.
///
/// `style_probs` is a list of (Move, probability) from the CNN.
/// `style_weight` controls how much the CNN influences the final score (default 250 centipawns).
pub fn find_best_move_hybrid(
    pos: &Chess,
    depth: u32,
    style_probs: &[(Move, f32)],
    style_weight: i32,
) -> Option<SearchResult> {
    if style_probs.is_empty() {
        return find_best_move(pos, depth);
    }

    let is_maximizing = pos.turn() == Color::White;

    let mut best_move = style_probs[0].0.clone();
    let mut best_score = if is_maximizing { i32::MIN } else { i32::MAX };

    for (mv, prob) in style_probs {
        let mut child = pos.clone();
        child.play_unchecked(mv);

        let tactical_score = minimax(&child, depth - 1, i32::MIN, i32::MAX, !is_maximizing);

        // Blend: tactical score + (style probability * weight)
        let style_bonus = (*prob * style_weight as f32) as i32;

        if is_maximizing {
            let final_score = tactical_score + style_bonus;
            if final_score > best_score {
                best_score = final_score;
                best_move = mv.clone();
            }
        } else {
            let final_score = tactical_score - style_bonus;
            if final_score < best_score {
                best_score = final_score;
                best_move = mv.clone();
            }
        }
    }

    Some(SearchResult {
        best_move,
        score: best_score,
    })
}
