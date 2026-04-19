/// SRK Chess Engine — Entry Point
///
/// UCI event loop + module declarations.

mod uci;
mod eval;
mod search;
mod nn;

use std::io::{self, BufRead, Write};
use uci::UciState;

fn main() {
    // Try to load the CNN model (optional — falls back to pure tactical if missing)
    let model = nn::ChessNN::load("model/chess_model.onnx")
        .ok()
        .or_else(|| {
            // Also check parent directory (if run from project root)
            nn::ChessNN::load("chess-engine-rs/model/chess_model.onnx").ok()
        });

    if model.is_some() {
        eprintln!("info string CNN model loaded successfully");
    } else {
        eprintln!("info string CNN model not found, using pure tactical mode");
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut state = UciState::new(model);

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if let Some(response) = state.handle_command(&line) {
            let mut out = stdout.lock();
            writeln!(out, "{}", response).ok();
            out.flush().ok();
        }
    }
}
