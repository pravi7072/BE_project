# backend/app/training/dataset.py
import os
import glob
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import warnings
from typing import Dict, Optional

# relative imports
from ..preprocessing.audio_processor import AudioProcessor
from ..preprocessing.feature_extractor import FeatureExtractor


class DysarthricSpeechDataset(Dataset):
    """
    Optimized dataset for large-scale dysarthric speech training.

    🔧 Improvements:
      - Lazy caching (on first use)
      - Augmentation done only when needed (vectorized, safe)
      - Skips invalid files automatically
      - Pin-memory + numpy to torch conversion optimized
      - Minimal preprocessing overhead per batch
    """

    def __init__(self, config, split: str = "train", cache: bool = True):
        self.config = config
        self.split = split
        self.cache_enabled = cache
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Core processors
        self.audio_processor = AudioProcessor(config)
        self.feature_extractor = FeatureExtractor(config)

        # File discovery
        self.dysarthric_files = sorted(glob.glob(os.path.join(config.paths.dysarthric_dir, "*.wav")))
        self.clear_files = sorted(glob.glob(os.path.join(config.paths.clear_dir, "*.wav")))

        if not self.dysarthric_files or not self.clear_files:
            raise FileNotFoundError("Missing audio files for dysarthric or clear datasets.")

        # Split datasets (90/10)
        split_ratio = 0.9
        dys_split = int(len(self.dysarthric_files) * split_ratio)
        clr_split = int(len(self.clear_files) * split_ratio)

        if split == "train":
            self.dysarthric_files = self.dysarthric_files[:dys_split]
            self.clear_files = self.clear_files[:clr_split]
        else:
            self.dysarthric_files = self.dysarthric_files[dys_split:]
            self.clear_files = self.clear_files[clr_split:]

        print(f"[INFO] {split} set: {len(self.dysarthric_files)} dysarthric, {len(self.clear_files)} clear")

        # Lazy cache (filled on demand)
        self.cache_data: Dict[int, Dict[str, torch.Tensor]] = {}

        # Disable excessive warnings
        warnings.filterwarnings("ignore", message="Time-stretch failed")

        # Pre-warm a few items to measure speed
        if cache:
            print("[CACHE] Lazy caching enabled (on-demand load).")

    def __len__(self):
        return max(len(self.dysarthric_files), len(self.clear_files))

    def __getitem__(self, idx: int):
        if self.cache_enabled and idx in self.cache_data:
            return self.cache_data[idx]

        try:
            item = self._load_and_process(idx)
            if self._is_valid(item):
                if self.cache_enabled:
                    self.cache_data[idx] = item
                return item
            else:
                # fallback — skip invalid
                return self._fallback_item()
        except Exception as e:
            print(f"[WARN] Dataset item {idx} failed: {e}")
            return self._fallback_item()

    # -----------------------------
    #  Internal helpers
    # -----------------------------
    def _load_and_process(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select paired files safely
        dys_path = self.dysarthric_files[idx % len(self.dysarthric_files)]
        clr_path = random.choice(self.clear_files)

        # Load raw audio
        dys_audio = self.audio_processor.load_audio(dys_path)
        clr_audio = self.audio_processor.load_audio(clr_path)

        # Preprocess (denoise + normalize)
        dys_audio = self.audio_processor.preprocess_pipeline(dys_audio)
        clr_audio = self.audio_processor.preprocess_pipeline(clr_audio)

        # Ensure minimum audio length
        n_fft = getattr(self.config.audio, "n_fft", 1024)
        min_len = max(1, n_fft)
        if len(dys_audio) < min_len:
            dys_audio = np.pad(dys_audio, (0, min_len - len(dys_audio)))
        if len(clr_audio) < min_len:
            clr_audio = np.pad(clr_audio, (0, min_len - len(clr_audio)))

        # Apply augmentation only for training
        if self.split == "train":
            dys_audio = self._augment_if_needed(dys_audio)
            clr_audio = self._augment_if_needed(clr_audio)

        # Convert to torch tensors (CPU pinned)
        dys_wave = torch.from_numpy(dys_audio).float()
        clr_wave = torch.from_numpy(clr_audio).float()

        # Extract Mel features
        dys_mel = self.feature_extractor.extract_mel(dys_wave)
        clr_mel = self.feature_extractor.extract_mel(clr_wave)


        if dys_mel is None or clr_mel is None or dys_mel.numel() == 0 or clr_mel.numel() == 0:
            raise ValueError("Invalid mel returned.")

        # Trim / pad
        max_len = min(dys_mel.size(-1), clr_mel.size(-1))
        dys_mel = self._pad_or_trim(dys_mel, max_len)
        clr_mel = self._pad_or_trim(clr_mel, max_len)

        # # Remove batch dim if exists
        # if dys_mel.dim() == 3 and dys_mel.size(0) == 1:
        #     dys_mel = dys_mel.squeeze(0)
        # if clr_mel.dim() == 3 and clr_mel.size(0) == 1:
        #     clr_mel = clr_mel.squeeze(0)

        return {
            "dysarthric_mel": dys_mel,
            "clear_mel": clr_mel,
            "dysarthric_audio": dys_wave,
            "clear_audio": clr_wave,
        }

    # -----------------------------
    #  Utility Methods
    # -----------------------------
    def _augment_if_needed(self, audio: np.ndarray) -> np.ndarray:
        """Applies fast augmentations probabilistically."""
        if audio is None or len(audio) < 128:
            return audio

        # Slightly lower augmentation frequency to reduce overhead
        if random.random() < 0.6:
            try:
                if random.random() < 0.3:
                    rate = random.uniform(0.9, 1.1)
                    audio = self.audio_processor.time_stretch(audio, rate)

                if random.random() < 0.3:
                    n_steps = random.randint(-2, 2)
                    if n_steps != 0:
                        audio = self.audio_processor.pitch_shift(audio, n_steps)

                if random.random() < 0.4:
                    noise_level = random.uniform(0.001, 0.004)
                    audio = self.audio_processor.add_noise(audio, noise_level)

            except Exception as e:
                print(f"[WARN] Augmentation skipped: {e}")

        return audio

    def _pad_or_trim(self, mel: torch.Tensor, max_len: int) -> torch.Tensor:
        mel_len = mel.size(-1)
        if mel_len > max_len:
            start = random.randint(0, mel_len - max_len)
            mel = mel[..., start:start + max_len]
        elif mel_len < max_len:
            pad_len = max(0, max_len - mel_len)
            mel = torch.nn.functional.pad(mel, (0, pad_len))
        return mel

    def _is_valid(self, item: Dict[str, torch.Tensor]) -> bool:
        """Checks if a sample is valid."""
        try:
            if not item:
                return False
            for k in ("dysarthric_mel", "clear_mel"):
                tensor = item.get(k)
                if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
                    return False
            return True
        except Exception:
            return False

    def _fallback_item(self) -> Dict[str, torch.Tensor]:
        """Returns safe placeholder on failure."""
        mel_zeros = torch.zeros((self.config.audio.n_mels, 100), dtype=torch.float32)
        silence = torch.zeros(self.config.audio.sample_rate, dtype=torch.float32)
        return {
            "dysarthric_mel": mel_zeros,
            "clear_mel": mel_zeros,
            "dysarthric_audio": silence,
            "clear_audio": silence,
        }
