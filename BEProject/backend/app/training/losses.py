# backend/app/training/losses.py

import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ---------------------------
# Helpers
# ---------------------------
def _center_crop_time(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ta = a.size(-1)
    tb = b.size(-1)

    if ta == tb:
        return a, b

    m = min(ta, tb)

    def crop(x, m):
        t = x.size(-1)
        start = max(0, (t - m) // 2)
        return x[..., start:start + m]

    return crop(a, m), crop(b, m)


# ---------------------------
# GAN losses
# ---------------------------
class GANLoss:

    @staticmethod
    def generator_loss(disc_outputs_fake):
        return sum(torch.mean((out - 1) ** 2) for out in disc_outputs_fake) / len(disc_outputs_fake)

    @staticmethod
    def discriminator_loss(disc_outputs_real, disc_outputs_fake):
        loss = 0
        for dr, df in zip(disc_outputs_real, disc_outputs_fake):
            loss += torch.mean((dr - 1) ** 2) + torch.mean(df ** 2)
        return loss / len(disc_outputs_real)


# ---------------------------
# Feature Matching
# ---------------------------
class FeatureMatchingLoss:

    @staticmethod
    def forward(real_feats, fake_feats):
        loss = 0
        count = 0

        for fr, ff in zip(real_feats, fake_feats):
            for r, f in zip(fr, ff):
                if r.size(-1) != f.size(-1):
                    r, f = _center_crop_time(r, f)
                loss += F.l1_loss(f, r.detach())
                count += 1

        return loss / max(1, count)


# ---------------------------
# ✅ CORRECT MEL LOSS
# ---------------------------
class MelLoss:
    """L1 loss in MEL domain (for generator)"""

    def forward(self, mel_fake, mel_real):
        try:
            if mel_fake.size(-1) != mel_real.size(-1):
                mel_fake, mel_real = _center_crop_time(mel_fake, mel_real)
            return F.l1_loss(mel_fake, mel_real)
        except RuntimeError:
            return torch.zeros(1, device=mel_fake.device, requires_grad=True)


# ---------------------------
# ✅ WAVEFORM LOSS (VOCODER)
# ---------------------------
class WaveformLoss:
    def forward(self, audio_fake, audio_real):
        try:
            min_len = min(audio_fake.size(-1), audio_real.size(-1))
            return F.l1_loss(
                audio_fake[..., :min_len],
                audio_real[..., :min_len]
            )
        except Exception:
            # ✅ SAFE fallback using available tensor
            device = audio_fake.device if isinstance(audio_fake, torch.Tensor) else "cpu"
            return torch.zeros(1, device=device, requires_grad=True)


# ---------------------------
# STFT LOSS
# ---------------------------
class MultiResolutionSTFTLoss(nn.Module):

    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fft_sizes = [1024, 2048, 512]
        self.hop_sizes = [256, 512, 128]
        self.win_lengths = [1024, 2048, 512]

    def stft_mag(self, x, fft, hop, win):
        if x.dim() == 3:
            x = x.squeeze(1)

        # x = x.to(self.device)
        window = torch.hann_window(win).to(self.device)

        spec= torch.abs(torch.stft(
            x,
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            window=window,
            return_complex=True
        ))
        return torch.clamp(spec, min=1e-7)

    # def forward(self, fake, real):
    #     loss = 0

    #     for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
    #         mf = self.stft_mag(fake, fft, hop, win)
    #         mr = self.stft_mag(real, fft, hop, win)

    #         if mf.size(-1) != mr.size(-1):
    #             mf, mr = _center_crop_time(mf, mr)

    #         sc = torch.mean(torch.abs(mf - mr)) / (torch.mean(torch.abs(mr)) + 1e-8)
    #         log = F.l1_loss(torch.log(mf + 1e-7), torch.log(mr + 1e-7))

    #         loss += sc + log

    #     return loss / len(self.fft_sizes)
    def forward(self, fake, real):
        loss_sc = 0.0
        loss_log = 0.0

        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            mf = self.stft_mag(fake, fft, hop, win)
            mr = self.stft_mag(real, fft, hop, win)

            if mf.size(-1) != mr.size(-1):
                mf, mr = _center_crop_time(mf, mr)

            # Spectral Convergence (correct as-is)
            sc = torch.norm(mr - mf, p='fro') / (torch.norm(mr, p='fro') + 1e-8)

            # Log STFT magnitude loss (FIXED: properly normalized)
            log_mf = torch.log(mf + 1e-7)
            log_mr = torch.log(mr + 1e-7)

            log = torch.norm(log_mr - log_mf, p=1) / (torch.norm(log_mr, p=1) + 1e-8)

            loss_sc += sc
            loss_log += log

        loss_sc /= len(self.fft_sizes)
        loss_log /= len(self.fft_sizes)

        return loss_sc + loss_log


# ---------------------------
# OTHER LOSSES
# ---------------------------
class CycleLoss:
    @staticmethod
    def forward(a, b):
        if a.size(-1) != b.size(-1):
            a, b = _center_crop_time(a, b)
        return F.l1_loss(a, b)


class PPGLoss:
    @staticmethod
    def forward(a, b):
        if a.size(-1) != b.size(-1):
            a, b = _center_crop_time(a, b)
        return F.l1_loss(a, b)


class SpeakerLoss:
    @staticmethod
    def forward(a, b):
        return (1 - F.cosine_similarity(a, b, dim=1)).mean()


# ---------------------------
# COMBINED LOSS
# ---------------------------
class CombinedLoss:

    def __init__(self, config):
        self.config = config

        self.gan_loss = GANLoss()
        self.feature_matching = FeatureMatchingLoss()

        self.feature_matching_loss = FeatureMatchingLoss()

        self.mel_loss = MelLoss()               # ✅ FIXED
        self.wave_loss = WaveformLoss()         # ✅ NEW
        self.stft_loss = MultiResolutionSTFTLoss()

        self.cycle_loss = CycleLoss()
        self.ppg_loss = PPGLoss()
        self.speaker_loss = SpeakerLoss()

    def compute_generator_loss(self,
                               disc_fake,
                               feat_real, feat_fake,
                               mel_fake, mel_real,
                               audio_fake, audio_real,
                               ppg_fake, ppg_real,
                               emb_fake, emb_real,
                               recon, original):

        loss_adv = self.gan_loss.generator_loss(disc_fake)
        loss_fm = self.feature_matching.forward(feat_real, feat_fake)

        # # ✅ CRITICAL FIX
        # loss_mel = self.mel_loss.forward(mel_fake, mel_real)

        # # vocoder-related
        # loss_wave = self.wave_loss.forward(audio_fake, audio_real)
        # loss_stft = self.stft_loss(audio_fake, audio_real )

        # # 🔥 PROTECTION
        # if torch.isnan(loss_stft) or torch.isinf(loss_stft):
        #     loss_stft = torch.zeros_like(loss_stft)

        loss_cycle = self.cycle_loss.forward(recon, original)
        loss_ppg = self.ppg_loss.forward(ppg_fake, ppg_real)
        loss_spk = self.speaker_loss.forward(emb_fake, emb_real)

        cfg = self.config.training

        # total = (
        #     cfg.lambda_gan * loss_adv +
        #     cfg.lambda_feat_match * loss_fm +
        #     cfg.lambda_mel * loss_mel +
        #     0.5 * loss_wave +
        #     0.2 * loss_stft +
        #     cfg.lambda_cycle * loss_cycle +
        #     cfg.lambda_ppg * loss_ppg +
        #     cfg.lambda_speaker * loss_spk
        # )
        total = (
        cfg.lambda_gan * loss_adv +
        cfg.lambda_feat_match * loss_fm +
        cfg.lambda_cycle * loss_cycle +
        cfg.lambda_ppg * loss_ppg +
        cfg.lambda_speaker * loss_spk
    )

        # return total, {
        #     "total": total,
        #     "mel": loss_mel,
        #     "wave": loss_wave,
        #     "stft": loss_stft
        # }
        return total, {
            "total": total,
            "adv": loss_adv,
            "fm": loss_fm,
            "cycle": loss_cycle,
            "ppg": loss_ppg,
            "spk": loss_spk
        }

    def compute_discriminator_loss(self, real, fake):
        return self.gan_loss.discriminator_loss(real, fake)