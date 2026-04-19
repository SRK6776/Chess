/// UCI (Universal Chess Interface) protocol handler.
///
/// Parses incoming commands and formats outgoing responses.
/// Reference: https://www.wbec-ridderkerk.nl/html/UCIProtocol.html

use shakmaty::{Chess, Position, uci::UciMove, CastlingMode, fen::Fen};
use crate::nn::ChessNN;

/// Holds the engine's internal state managed by UCI commands.
pub struct UciState {
    pub position: Chess,
    pub debug: bool,
    pub model: Option<ChessNN>,
    pub style_weight: i32,
}

impl UciState {
    pub fn new(model: Option<ChessNN>) -> Self {
        UciState {
            position: Chess::default(),
            debug: false,
            model,
            style_weight: 250, // Default style weight (250 centipawns = 2.5 in Python)
        }
    }

    /// Processes a single UCI command string and returns a response (if any).
    pub fn handle_command(&mut self, input: &str) -> Option<String> {
        let input = input.trim();
        if input.is_empty() {
            return None;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts[0];

        match command {
            "uci" => Some(self.cmd_uci()),
            "isready" => Some("readyok".to_string()),
            "ucinewgame" => { self.cmd_new_game(); None }
            "position" => { self.cmd_position(&parts); None }
            "go" => Some(self.cmd_go(&parts)),
            "debug" => {
                if parts.len() > 1 {
                    self.debug = parts[1] == "on";
                }
                None
            }
            "quit" => std::process::exit(0),
            _ => {
                if self.debug {
                    Some(format!("info string unknown command: {}", command))
                } else {
                    None
                }
            }
        }
    }

    /// Responds to the `uci` command with engine identification.
    fn cmd_uci(&self) -> String {
        let mut response = String::new();
        response.push_str("id name SRK-Chess-Engine 0.1\n");
        response.push_str("id author SarthakKale\n");
        response.push_str("option name StyleWeight type spin default 250 min 0 max 1000\n");
        response.push_str("uciok");
        response
    }

    /// Resets the engine for a new game.
    fn cmd_new_game(&mut self) {
        self.position = Chess::default();
    }

    /// Parses a `position` command and updates the internal board.
    fn cmd_position(&mut self, parts: &[&str]) {
        if parts.len() < 2 {
            return;
        }

        let mut move_start_idx: Option<usize> = None;

        if parts[1] == "startpos" {
            self.position = Chess::default();
            if parts.len() > 2 && parts[2] == "moves" {
                move_start_idx = Some(3);
            }
        } else if parts[1] == "fen" {
            let mut fen_end = 2;
            for i in 2..parts.len() {
                if parts[i] == "moves" {
                    move_start_idx = Some(i + 1);
                    break;
                }
                fen_end = i + 1;
            }
            let fen_str = parts[2..fen_end].join(" ");
            if let Ok(fen) = fen_str.parse::<Fen>() {
                if let Ok(pos) = fen.into_position::<Chess>(CastlingMode::Standard) {
                    self.position = pos;
                }
            }
        }

        // Apply moves if present
        if let Some(start) = move_start_idx {
            for i in start..parts.len() {
                if let Ok(uci_move) = parts[i].parse::<UciMove>() {
                    if let Ok(m) = uci_move.to_move(&self.position) {
                        self.position.play_unchecked(&m);
                    }
                }
            }
        }
    }

    /// Handles the `go` command. Parses depth and runs hybrid or pure search.
    fn cmd_go(&mut self, parts: &[&str]) -> String {
        // Parse depth (default = 5)
        let mut depth: u32 = 5;
        for i in 0..parts.len() {
            if parts[i] == "depth" {
                if let Some(d) = parts.get(i + 1) {
                    if let Ok(d) = d.parse::<u32>() {
                        depth = d;
                    }
                }
            }
        }

        // Try hybrid search (with CNN), fallback to pure tactical
        let result = if let Some(ref mut model) = self.model {
            let style_probs = model.get_style_probs(&self.position);
            crate::search::find_best_move_hybrid(
                &self.position,
                depth,
                &style_probs,
                self.style_weight,
            )
        } else {
            crate::search::find_best_move(&self.position, depth)
        };

        match result {
            Some(result) => {
                let uci_move = UciMove::from_standard(&result.best_move);
                let score_cp = result.score;
                let mode = if self.model.is_some() { "hybrid" } else { "tactical" };
                format!(
                    "info depth {} score cp {} string mode={}\nbestmove {}",
                    depth, score_cp, mode, uci_move
                )
            }
            None => "bestmove 0000".to_string(),
        }
    }
}
