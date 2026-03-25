import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional


class FeatureExtractor:
    """
    🚀 Production-ready feature extractor (optimized for stability).

    FINAL NOTES:
    - Uses CPU to avoid GPU OOM
    - Produces LOG-MEL (only once)
    - Fully aligned with training + inference
    """

    def __init__(self, config):
        self.config = config

        # ✅ KEEP CPU (important for your setup)
        self.device = torch.device("cpu")

        # ---- Audio / Feature Parameters ----
        self.sample_rate = config.audio.sample_rate
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.win_length = config.audio.win_length
        self.n_mels = config.audio.n_mels
        self.n_mfcc = config.audio.n_mfcc
        self.fmin = getattr(config.audio, "fmin", 0)
        self.fmax = getattr(config.audio, "fmax", self.sample_rate // 2)

        # ---- Cached transforms ----
        self._init_transforms()

        # Dummy fallback
        self._dummy_audio = torch.zeros(1, self.sample_rate)
        self._dummy_mel = torch.zeros(1, self.n_mels, 100)

    # ==========================================================
    # 🔧 TRANSFORMS
    # ==========================================================
    def _init_transforms(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0,
            normalized=False,
        )

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "f_min": self.fmin,
                "f_max": self.fmax,
            },
        )

        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        )

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_iter=32,
            power=2.0,
        )

    # ==========================================================
    # 🎵 MEL SPECTROGRAM (CORRECT)
    # ==========================================================
    @torch.inference_mode()
    def extract_mel(self, audio: torch.Tensor, return_db: bool = True) -> torch.Tensor:
        import librosa
        import numpy as np

        try:
            # -----------------------------
            # Ensure tensor format
            # -----------------------------
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio.dim() == 2:
                audio = audio.squeeze(0)

            if audio.numel() == 0:
                return self._dummy_mel.clone()

            # -----------------------------
            # Convert to numpy
            # -----------------------------
            audio_np = audio.cpu().numpy()

            # -----------------------------
            # MEL SPECTROGRAM (HiFi-GAN MATCH)
            # -----------------------------
            mel = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=2.0   # ✅ CRITICAL (HiFi-GAN standard)
            )

            # -----------------------------
            # LOG SCALE (NATURAL LOG)
            # -----------------------------
            mel = np.log(np.clip(mel, 1e-5, None))
            mel = np.clip(mel, -11.5, 2.0)

            # -----------------------------
            # Convert back to tensor
            # -----------------------------
            mel = torch.from_numpy(mel).unsqueeze(0).float()

            # -----------------------------
            # Safety (NaN / Inf handling)
            # -----------------------------
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)

            return mel

        except Exception as e:
            print(f"[ERROR] extract_mel failed: {e}")
            return self._dummy_mel.clone()

    # ==========================================================
    # 🎵 MFCC
    # ==========================================================
    @torch.inference_mode()
    def extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            if audio.numel() == 0:
                return torch.zeros((1, self.n_mfcc, 100))

            audio = audio.to("cpu")
            mfcc = self.mfcc_transform(audio)
            mfcc = torch.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0)
            return mfcc.cpu()

        except Exception as e:
            print(f"[ERROR] extract_mfcc failed: {e}")
            return torch.zeros((1, self.n_mfcc, 100))

    # ==========================================================
    # 🔄 MEL → AUDIO
    # ==========================================================
    @torch.inference_mode()
    def mel_to_audio(self, mel: torch.Tensor, use_griffin_lim: bool = True) -> torch.Tensor:
        try:
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            mel = mel.to("cpu")

            # NOTE: Not critical for your pipeline (vocoder used instead)
            # mel = torchaudio.functional.DB_to_amplitude(mel, ref=1.0, power=0.5)
            mel = torch.exp(mel)
            mel = torch.nan_to_num(mel)

            if use_griffin_lim:
                spec = self.inverse_mel(mel)
                audio = self.griffin_lim(spec)
            else:
                mel_np = mel.squeeze(0).cpu().numpy()
                audio_np = librosa.feature.inverse.mel_to_audio(
                    mel_np,
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                )
                audio = torch.from_numpy(audio_np).float().unsqueeze(0)

            audio = torch.clamp(audio, -1.0, 1.0)
            return audio.cpu()

        except Exception as e:
            print(f"[WARN] mel_to_audio failed: {e}")
            return self._dummy_audio.cpu()

    # ==========================================================
    # 📈 STFT
    # ==========================================================
    @torch.inference_mode()
    def compute_stft(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()

            if audio.dim() == 1:
                audio = audio.unsqueeze(0)

            if audio.numel() == 0:
                raise ValueError("Empty input")

            audio = audio.to("cpu")

            stft = torch.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                return_complex=True,
            )

            mag = torch.abs(stft)
            phase = torch.angle(stft)

            return mag.cpu(), phase.cpu()

        except Exception as e:
            print(f"[ERROR] compute_stft failed: {e}")
            dummy = torch.zeros((1, self.n_fft // 2 + 1, 10))
            return dummy, dummy

    # ==========================================================
    # ⚖️ NORMALIZATION
    # ==========================================================
    @staticmethod
    def normalize_mel(mel: torch.Tensor, mean: Optional[float] = None, std: Optional[float] = None) -> torch.Tensor:
        mel = mel.to(torch.float32)
        if mel.numel() == 0:
            return mel
        mean = mel.mean() if mean is None else mean
        std = mel.std() + 1e-8 if std is None or std == 0 else std
        return torch.nan_to_num((mel - mean) / std)

    @staticmethod
    def denormalize_mel(mel: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        mel = mel.to(torch.float32)
        return torch.nan_to_num(mel * std + mean)
