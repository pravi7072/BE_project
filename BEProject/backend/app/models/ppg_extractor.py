# backend/app/models/ppg_extractor.py
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Optional


class PPGExtractor(nn.Module):
    """Production PPG extractor using pre-trained Wav2Vec2 (SAFE VERSION)"""

    def __init__(self, config, pretrained_model: str = "facebook/wav2vec2-base-960h"):
        super().__init__()
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pretrained model
        self.processor = Wav2Vec2Processor.from_pretrained(pretrained_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_model).to(self.device)

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        hidden_size = self.model.config.hidden_size

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, config.model.ppg_dim * 2),
            nn.ReLU(),
            nn.Linear(config.model.ppg_dim * 2, config.model.ppg_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, T) waveform ONLY
        Returns:
            ppg: (B, ppg_dim, T')
        """

        if audio.dim() == 3:
            # ❌ Prevent silent bug
            raise ValueError("PPGExtractor expects raw waveform, not mel")

        # Ensure float + device
        audio = audio.to(self.device).float()

        # Normalize (important for wav2vec2)
        audio = torch.clamp(audio, -1.0, 1.0)

        with torch.no_grad():
            outputs = self.model(
                audio,
                output_hidden_states=False,
                return_dict=True
            )

            hidden_states = outputs.logits  # safer than wav2vec2 internal call

        # Project to PPG
        ppg = self.projection(hidden_states)  # (B, T', ppg_dim)
        ppg = ppg.transpose(1, 2)             # (B, ppg_dim, T')

        return ppg


# ==========================================================
# ✅ TRAINING PPG EXTRACTOR (USE THIS)
# ==========================================================
class SimplePPGExtractor(nn.Module):
    """Lightweight PPG extractor for training (CORRECT)"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(config.audio.n_mels, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, config.model.ppg_dim, kernel_size=1)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T)
        Returns:
            ppg: (B, ppg_dim, T)
        """
        ppg = self.conv_blocks(mel)
        ppg = torch.tanh(ppg)
        return ppg