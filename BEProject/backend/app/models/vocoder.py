# backend/app/models/vocoder.py
from backend.app.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ResBlock(nn.Module):
    """Residual block for HiFi-GAN"""
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, 
                     dilation=d, padding=self.get_padding(kernel_size, d))
            for d in dilations
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     dilation=1, padding=self.get_padding(kernel_size, 1))
            for _ in dilations
        ])
    
    @staticmethod
    def get_padding(kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class MRF(nn.Module):
    """Multi-Receptive Field Fusion"""
    def __init__(self, channels, kernel_sizes=[3, 7, 11], 
                 dilations_list=[[1,3,5], [1,3,5], [1,3,5]]):
        super().__init__()
        
        self.resblocks = nn.ModuleList([
            ResBlock(channels, k, d) 
            for k, d in zip(kernel_sizes, dilations_list)
        ])
    
    def forward(self, x):
        output = None
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output = output + resblock(x)
        return output / len(self.resblocks)

class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator for high-quality audio synthesis"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Configuration
        upsample_rates = config.model.vocoder_upsample_rates
        upsample_kernel_sizes = config.model.vocoder_upsample_kernel_sizes
        resblock_kernel_sizes = config.model.vocoder_resblock_kernel_sizes
        resblock_dilations = config.model.vocoder_resblock_dilation_sizes
        
        initial_channels = 512
        
        # Pre-convolution
        self.pre_conv = nn.Conv1d(config.audio.n_mels, initial_channels, 7, 1, padding=3)
        
        # Upsampling blocks
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        channels = initial_channels
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    channels, channels // 2, kernel_size, rate,
                    padding=(kernel_size - rate) // 2
                )
            )
            
            self.mrfs.append(
                MRF(channels // 2, resblock_kernel_sizes, resblock_dilations)
            )
            
            channels = channels // 2
        
        # Post-convolution
        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, 1, 7, 1, padding=3),
            nn.Tanh()
        )
    
    def forward(self, mel, speaker_emb=None):
        """
        Args:
            mel: (B, n_mels, T)
            speaker_emb: Optional speaker embedding (not used in basic HiFi-GAN)
        Returns:
            audio: (B, 1, T * prod(upsample_rates))
        """
        x = self.pre_conv(mel)
        
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        
        audio = self.post_conv(x)
        
        return audio
    
    def inference(self, mel):
        """Inference without gradient computation"""
        with torch.no_grad():
            return self.forward(mel)


class HiFiGANDiscriminator(nn.Module):
    """Multi-scale + Multi-period discriminators for HiFi-GAN"""
    def __init__(self):
        super().__init__()
        self.msd = MultiScaleDiscriminatorVocoder()
        self.mpd = MultiPeriodDiscriminator()
    
    def forward(self, x):
        msd_outputs, msd_features = self.msd(x)
        mpd_outputs, mpd_features = self.mpd(x)
        
        return msd_outputs + mpd_outputs, msd_features + mpd_features

class MultiScaleDiscriminatorVocoder(nn.Module):
    """Multi-scale discriminator for vocoder"""
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminatorVocoder() for _ in range(3)
        ])
        self.pooling = nn.AvgPool1d(4, 2, padding=2)
    
    def forward(self, x):
        outputs = []
        features = []
        
        for i, disc in enumerate(self.discriminators):
            if i != 0:
                x = self.pooling(x)
            
            out, feat = disc(x)
            outputs.append(out)
            features.append(feat)
        
        return outputs, features

class ScaleDiscriminatorVocoder(nn.Module):
    """Single scale discriminator for vocoder"""
    def __init__(self):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ])
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(1, 64, 15, 1, padding=7),
        #     nn.Conv1d(64, 64, 41, 2, groups=4, padding=20),
        #     nn.Conv1d(64, 128, 41, 2, groups=8, padding=20),
        #     nn.Conv1d(128, 256, 41, 4, groups=8, padding=20),
        #     nn.Conv1d(256, 512, 41, 4, groups=8, padding=20),
        #     nn.Conv1d(512, 512, 41, 1, groups=8, padding=20),
        #     nn.Conv1d(512, 512, 5, 1, padding=2),
        # ])
        self.output = nn.Conv1d(1024, 1, 3, 1, padding=1)
        # self.output = nn.Conv1d(512, 1, 3, 1, padding=1)
    
    def forward(self, x):
        features = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        
        x = self.output(x)
        features.append(x)
        
        return x, features
