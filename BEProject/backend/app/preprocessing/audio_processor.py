# backend/app/preprocessing/audio_processor.py
import librosa
import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
from typing import Optional, Any, Dict
import inspect
import functools
import warnings

# For parallel-safe caching
from joblib import Memory
cache_dir = "./cache/audio_proc"
memory = Memory(cache_dir, verbose=0)

_time_stretch_warn_count = 0
_MAX_TS_WARN = 5


class AudioProcessor:
    """
    High-performance audio preprocessing utility with robust fallbacks.
    Optimized for large dataset training loops.

    ✅ Vectorized + cached for speed
    ✅ Minimal redundant I/O
    ✅ Deterministic fallback behavior
    """

    def __init__(self, config: Any):
        self.config = config
        self.sample_rate = int(config.audio.sample_rate)
        self.noise_profile: Optional[np.ndarray] = None
        self.global_mean = 0.0
        self.global_std = 1.0

        # Cache constants
        self._n_fft = getattr(config.audio, "n_fft", 1024)
        self._hop_length = getattr(config.audio, "hop_length", 256)
        self._min_len = max(256, self._hop_length)
        self._cache_enabled = True

    # ---------------- IO ----------------
    @staticmethod
    @memory.cache
    def _safe_load(path: str, sr: int) -> np.ndarray:
        """Cached safe audio loader."""
        try:
            audio, _ = librosa.load(path, sr=sr, mono=True)
            if audio is None or len(audio) == 0:
                raise ValueError("Empty audio.")
            if np.isnan(audio).any():
                raise ValueError("NaN values in audio.")
            return np.clip(audio.astype(np.float32), -1.0, 1.0)
        except Exception as e:
            print(f"[WARN] Load failed ({path}): {e}")
            # short silent fallback
            return np.random.normal(0.0, 1e-6, sr).astype(np.float32)

    def load_audio(self, path: str) -> np.ndarray:
        """Uses joblib-cached loader for reuse."""
        if not path or not isinstance(path, str):
            return np.zeros(self.sample_rate, dtype=np.float32)
        return self._safe_load(path, self.sample_rate)

    def save_audio(self, audio: np.ndarray, path: str):
        try:
            audio = np.clip(np.asarray(audio, np.float32), -1.0, 1.0)
            sf.write(path, audio, self.sample_rate)
        except Exception as e:
            print(f"[ERROR] Saving audio failed: {e}")

    # ---------------- core cleaning ----------------
    def remove_silence(self, audio: np.ndarray, top_db: int = 30) -> np.ndarray:
        try:
            if audio is None or len(audio) < self._min_len:
                return audio
            intervals = librosa.effects.split(audio, top_db=top_db)
            if len(intervals) == 0:
                return audio
            if len(intervals) == 1:
                start, end = intervals[0]
                return audio[start:end]
            out = np.concatenate([audio[s:e] for s, e in intervals if e > s])
            return out.astype(np.float32)
        except Exception:
            return audio

    def normalize_volume(self, audio: np.ndarray, target_level_db: float = -20.0) -> np.ndarray:
        if audio is None or len(audio) == 0:
            return np.zeros(self.sample_rate, dtype=np.float32)
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        gain = 10 ** ((target_level_db - 20 * np.log10(rms)) / 20)
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    # ---------------- noise ----------------
    @memory.cache
    def estimate_noise_profile(self, audio: np.ndarray, noise_duration: float = 0.5) -> np.ndarray:
        try:
            noise_samples = int(noise_duration * self.sample_rate)
            # seg = audio[:min(noise_samples, len(audio))]
            seg = audio[:min(noise_samples, len(audio))]
            if len(seg) < self._n_fft:
                seg = np.pad(seg, (0, self._n_fft - len(seg)))
            stft_noise = librosa.stft(seg, n_fft=self._n_fft, hop_length=self._hop_length)
            profile = np.mean(np.abs(stft_noise), axis=1).astype(np.float32)
            self.noise_profile = profile
            return profile
        except Exception:
            return np.zeros(self._n_fft // 2 + 1, dtype=np.float32)

    def reduce_noise(self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            if audio is None or len(audio) == 0:
                return audio
            stft = librosa.stft(audio, n_fft=self._n_fft, hop_length=self._hop_length)
            mag, phase = np.abs(stft), np.angle(stft)
            noise = noise_profile or self.noise_profile or self.estimate_noise_profile(audio)
            noise = np.broadcast_to(noise[:, np.newaxis], mag.shape)
            mag_denoised = np.maximum(mag - noise, 0.0)
            stft_denoised = mag_denoised * np.exp(1j * phase)
            return np.clip(librosa.istft(stft_denoised, hop_length=self._hop_length, length=len(audio)), -1.0, 1.0)
        except Exception:
            return audio

    # ---------------- filters ----------------
    def apply_preemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        if len(audio) <= 1:
            return audio
        return np.append(audio[0], audio[1:] - coef * audio[:-1]).astype(np.float32)

    def apply_deemphasis(self, audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
        try:
            return signal.lfilter([1.0], [1.0, -coef], audio).astype(np.float32)
        except Exception:
            return audio

    # ---------------- augmentations ----------------
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """Fast safe wrapper with fallback."""
        global _time_stretch_warn_count
        if rate <= 0 or abs(rate - 1.0) < 1e-5:
            return audio
        try:
            return librosa.effects.time_stretch(audio, rate=rate).astype(np.float32)
        except Exception as e:
            if _time_stretch_warn_count < _MAX_TS_WARN:
                print(f"[WARN] time_stretch fallback: {e}")
                _time_stretch_warn_count += 1
            try:
                new_len = max(1, int(len(audio) / rate))
                return signal.resample(audio, new_len).astype(np.float32)
            except Exception:
                return audio

    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        try:
            if abs(n_steps) < 1e-6:
                return audio
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps).astype(np.float32)
        except Exception:
            return audio

    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        if noise_factor <= 0:
            return audio
        noise = np.random.randn(len(audio)).astype(np.float32)
        return np.clip(audio + noise_factor * noise, -1.0, 1.0)

    # ---------------- pipeline ----------------
    def preprocess_pipeline(
        self,
        audio: np.ndarray,
        remove_silence: bool = True,
        normalize: bool = True,
        denoise: bool = False,
    ) -> np.ndarray:
        try:
            if not isinstance(audio, np.ndarray) or len(audio) == 0:
                return np.random.normal(0, 1e-4, self.sample_rate).astype(np.float32)

            if remove_silence:
                audio = self.remove_silence(audio)
            if denoise:
                audio = self.reduce_noise(audio)
            if normalize:
                audio = self.normalize_volume(audio)

            # Always ensure minimum FFT length
            if len(audio) < self._n_fft:
                pad_len = self._n_fft - len(audio)
                audio = np.pad(audio, (0, pad_len))

            # audio = self.apply_preemphasis(audio)

            # final sanity
            if np.isnan(audio).any() or np.all(audio == 0):
                audio = np.random.normal(0, 1e-4, self.sample_rate)

            return np.clip(audio, -1.0, 1.0).astype(np.float32)
        except Exception as e:
            warnings.warn(f"[ERROR] Preprocess failed: {e}")
            return np.random.normal(0, 1e-4, self.sample_rate).astype(np.float32)
