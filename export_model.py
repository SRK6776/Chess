"""
Exports the trained PyTorch chess CNN to ONNX format
for native inference in the Rust engine.

Usage: python export_model.py
Output: chess-engine-rs/model/chess_model.onnx
"""

import torch
import os
import sys

# Add parent directory so we can import ChessCNN
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import ChessCNN

def export():
    model_path = "chess_model.pth"
    output_dir = "chess-engine-rs/model"
    output_path = os.path.join(output_dir, "chess_model.onnx")

    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Train the model first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = ChessCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Dummy input matching training shape: (batch=1, channels=12, height=8, width=8)
    dummy_input = torch.randn(1, 12, 8, 8)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board"],
        output_names=["move_logits"],
        dynamic_axes={
            "board": {0: "batch"},
            "move_logits": {0: "batch"},
        },
        opset_version=17,
    )

    # Verify the file was created
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported to {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    export()
