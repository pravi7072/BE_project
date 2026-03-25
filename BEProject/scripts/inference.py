# scripts/inference.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from pathlib import Path
import numpy as np

from backend.app.utils.config import Config
from backend.app.models.model_manager import ModelManager
from backend.app.preprocessing.audio_processor import AudioProcessor
from backend.app.preprocessing.feature_extractor import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description='Inference for Dysarthric Speech Conversion')
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch')
    args = parser.parse_args()

    # -------------------------------
    # Initialize config (STABLE MODE)
    # -------------------------------
    config = Config()
    config.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Disable unstable inference tricks
    config.use_half_precision = False
    config.use_quantization = False

    print("CUDA available:", torch.cuda.is_available())
    print("Using device:", config.device)

    # -------------------------------
    # Load models
    # -------------------------------
    print("Loading models...")
    model_manager = ModelManager(config, args.checkpoint)
    print("Models loaded successfully!")
    print(model_manager.get_model_info())

    # -------------------------------
    # Initialize processors
    # -------------------------------
    audio_processor = AudioProcessor(config)
    feature_extractor = FeatureExtractor(config)

    # -------------------------------
    # Output directory
    # -------------------------------
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------
    # Input files
    # -------------------------------
    input_path = Path(args.input)
    if input_path.is_file():
        audio_files = [input_path]
    elif input_path.is_dir():
        audio_files = list(input_path.glob("*.wav"))
        audio_files.extend(input_path.glob("*.mp3"))
    else:
        raise ValueError(f"Invalid input path: {args.input}")

    print(f"Found {len(audio_files)} audio files to process")

    # -------------------------------
    # Inference loop
    # -------------------------------
    for audio_file in audio_files:
        print(f"\nProcessing: {audio_file.name}")

        try:
            # ---------------------------
            # Load & preprocess
            # ---------------------------
            audio = audio_processor.preprocess_pipeline(
                audio_processor.load_audio(str(audio_file))
            )

            # ---------------------------
            # Extract mel
            # ---------------------------
            mel = feature_extractor.extract_mel(torch.FloatTensor(audio))


            # Ensure shape
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            if mel.dim() == 3 and mel.size(0) != 1:
                mel = mel[:1]

            mel = mel.to(config.device)
            print("Mel min/max:", mel.min().item(), mel.max().item())

            print("Input mel shape:", mel.shape, "| device:", mel.device)

            # ---------------------------
            # Convert
            # ---------------------------
            print("Converting...")
            with torch.no_grad():
                audio_clear = model_manager.convert(mel)

            # ---------------------------
            # Validate output
            # ---------------------------
            if audio_clear is None or audio_clear.numel() == 0:
                raise ValueError("Model returned empty output")

            print("Generated audio shape:", audio_clear.shape)

            # ---------------------------
            # Post-processing
            # ---------------------------
            audio_clear_np = audio_clear.squeeze().detach().cpu().numpy()

            # Remove NaNs/Infs
            audio_clear_np = np.nan_to_num(audio_clear_np)

            # Safe normalization
            max_val = np.max(np.abs(audio_clear_np))
            audio_clear_np = audio_clear_np / (max_val + 1e-6)

            # De-emphasis
            audio_clear_np = audio_processor.apply_deemphasis(audio_clear_np)

            # ---------------------------
            # Save output
            # ---------------------------
            output_file = output_path / f"{audio_file.stem}_clear.wav"
            audio_processor.save_audio(audio_clear_np, str(output_file))

            print(f"Saved: {output_file.name}")

        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\nInference completed!")


if __name__ == "__main__":
    main()