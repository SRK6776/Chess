/// Neural Network Module — ONNX CNN Inference
///
/// Loads the exported chess_model.onnx and runs inference natively.
/// Replaces the PyTorch inference from ai_player.py with zero Python dependency.
///
/// Input:  12x8x8 one-hot tensor (same format as Python training)
/// Output: 4096 move logits (from_sq * 64 + to_sq)

use ndarray::Array4;
use ort::session::Session;
use shakmaty::{Board, Chess, Color, Move, Piece, Position, Role};

/// Wraps the ONNX model session and provides inference methods.
pub struct ChessNN {
    session: Session,
}

impl ChessNN {
    /// Loads the ONNX model from disk.
    pub fn load(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        Ok(ChessNN { session })
    }

    /// Converts a shakmaty Board into the 12x8x8 tensor format the CNN expects.
    fn board_to_tensor(board: &Board) -> Array4<f32> {
        let mut tensor = Array4::<f32>::zeros((1, 12, 8, 8));

        let piece_to_layer = |piece: Piece| -> usize {
            let base = match piece.role {
                Role::Pawn   => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook   => 3,
                Role::Queen  => 4,
                Role::King   => 5,
            };
            match piece.color {
                Color::White => base,
                Color::Black => base + 6,
            }
        };

        for sq in board.occupied() {
            if let Some(piece) = board.piece_at(sq) {
                let layer = piece_to_layer(piece);
                let row = 7 - sq.rank() as usize;
                let col = sq.file() as usize;
                tensor[[0, layer, row, col]] = 1.0;
            }
        }

        tensor
    }

    /// Maps a shakmaty Move to the 0-4095 index space (from_sq * 64 + to_sq).
    fn move_to_index(mv: &Move) -> usize {
        let from_sq = match mv {
            Move::Normal { from, .. } => *from,
            Move::EnPassant { from, .. } => *from,
            Move::Castle { king, .. } => *king,
            Move::Put { .. } => return 0,
        };
        let to_sq = mv.to();

        let from_idx = from_sq.file() as usize + (from_sq.rank() as usize) * 8;
        let to_idx = to_sq.file() as usize + (to_sq.rank() as usize) * 8;

        from_idx * 64 + to_idx
    }

    /// Runs the CNN on the current position and returns softmax probabilities
    /// for each legal move.
    pub fn get_style_probs(&mut self, pos: &Chess) -> Vec<(Move, f32)> {
        let tensor = Self::board_to_tensor(pos.board());

        // Run ONNX inference — pass owned array
        let input_value = match ort::value::Value::from_array(tensor) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let outputs = match self.session.run(ort::inputs![input_value]) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        // Extract logits: shape [1, 4096]
        let logits_data: &[f32] = match outputs[0].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => data,
            Err(_) => return Vec::new(),
        };

        // Softmax over legal move indices only
        let legals = pos.legal_moves();
        let mut move_logits: Vec<(Move, f32)> = Vec::with_capacity(legals.len());

        let mut max_logit = f32::NEG_INFINITY;
        for mv in legals.iter() {
            let idx = Self::move_to_index(mv);
            let logit = if idx < logits_data.len() { logits_data[idx] } else { 0.0 };
            max_logit = max_logit.max(logit);
            move_logits.push((mv.clone(), logit));
        }

        // Stable softmax
        let mut sum_exp = 0.0f32;
        for (_, logit) in move_logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum_exp += *logit;
        }
        for (_, prob) in move_logits.iter_mut() {
            *prob /= sum_exp;
        }

        move_logits
    }
}
