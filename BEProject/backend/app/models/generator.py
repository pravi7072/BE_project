# backend/app/models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.app.models.discriminator import SafeInstanceNorm1d


class AdaptiveInstanceNorm(nn.Module):
    """Adaptive Instance Normalization for style transfer (STABLE)"""
    def __init__(self, num_features, style_dim):
        super().__init__()

        # ✅ FIX: use SafeInstanceNorm
        self.norm = SafeInstanceNorm1d(num_features)

        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, style):
        h = self.fc(style)
        h = h.view(h.size(0), h.size(1), 1)

        gamma, beta = torch.chunk(h, 2, dim=1)

        return (1 + gamma) * self.norm(x) + beta


class ResidualBlock(nn.Module):
    """Residual block with AdaIN"""
    def __init__(self, channels, style_dim, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)

        self.adain1 = AdaptiveInstanceNorm(channels, style_dim)
        self.adain2 = AdaptiveInstanceNorm(channels, style_dim)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, style):
        residual = x

        out = self.conv1(x)
        out = self.adain1(out, style)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.adain2(out, style)

        return self.relu(out + residual)


class Generator(nn.Module):
    """Advanced Generator with style modulation"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        ppg_dim = config.model.ppg_dim
        speaker_dim = config.model.speaker_emb_dim
        mel_dim = config.audio.n_mels
        channels = config.model.generator_channels
        n_res_blocks = config.model.n_res_blocks

        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(speaker_dim, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

        # =========================
        # INPUT
        # =========================
        self.input_conv = nn.Sequential(
            nn.Conv1d(ppg_dim, channels, 7, padding=3),

            # ✅ FIX
            SafeInstanceNorm1d(channels),

            nn.ReLU()
        )

        # =========================
        # DOWNSAMPLING
        # =========================
        self.down1 = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 3, stride=2, padding=1),

            # ✅ FIX
            SafeInstanceNorm1d(channels * 2),

            nn.ReLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(channels * 2, channels * 4, 3, stride=2, padding=1),

            # ✅ FIX
            SafeInstanceNorm1d(channels * 4),

            nn.ReLU()
        )

        # =========================
        # RESIDUAL BLOCKS
        # =========================
        dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8]

        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels * 4, channels, dilation=dilations[i % len(dilations)])
            for i in range(n_res_blocks)
        ])

        # =========================
        # UPSAMPLING
        # =========================
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(channels * 4, channels * 2, 4, stride=2, padding=1),

            # ✅ FIX
            SafeInstanceNorm1d(channels * 2),

            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(channels * 2, channels, 4, stride=2, padding=1),

            # ✅ FIX
            SafeInstanceNorm1d(channels),

            nn.ReLU()
        )

        # =========================
        # OUTPUT
        # =========================
        self.output_conv = nn.Conv1d(channels, mel_dim, 7, padding=3)

    def forward(self, ppg: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ppg: (B, ppg_dim, T)
            speaker_emb: (B, speaker_dim)
        Returns:
            mel: (B, mel_dim, T)
        """

        # Style
        style = self.style_encoder(speaker_emb)

        # Input
        x = self.input_conv(ppg)

        # Downsample
        x = self.down1(x)
        x = self.down2(x)

        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x, style)

        # Upsample
        x = self.up1(x)
        x = self.up2(x)

        # Output
        mel = self.output_conv(x)
        # Optional mild stabilization (SAFE)
        mel = torch.tanh(mel) * 7.0 - 5.0
        # mel = torch.tanh(mel) * 6.0 - 6.0

        return mel
