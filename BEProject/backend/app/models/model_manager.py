# backend/app/models/model_manager.py
import torch
import torch.nn as nn
import os
from typing import Optional, Dict
import numpy as np
from backend.app.models.pretrained_vocoder import PretrainedHiFiGAN
class ModelManager:
    """Manages model loading, optimization, and inference"""
    
    def __init__(self, config, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = config.device
        self.use_pretrained_vocoder = True
        # Initialize models
        self._init_models()
        
        # Load checkpoint
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Optimize models for inference
        self._optimize_models()
    
    def _init_models(self):
        """Initialize models for inference"""
        from .generator import Generator
        from .ppg_extractor import SimplePPGExtractor
        from .speaker_encoder import SpeakerEncoder
        from .vocoder import HiFiGANGenerator
        
        print("Initializing inference models...")
        
        # Generator (only I2C for dysarthric to clear)
        self.generator = Generator(self.config).to(self.device)
        
        # PPG extractor
        self.ppg_extractor = SimplePPGExtractor(self.config).to(self.device)
        
        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(self.config).to(self.device)
        
        # Vocoder
        # self.vocoder = HiFiGANGenerator(self.config).to(self.device)
        # Vocoder (switchable)
        if self.use_pretrained_vocoder:
            print("✅ Using PRETRAINED vocoder")

            self.vocoder = PretrainedHiFiGAN(
                checkpoint_path="pretrained/generator_v1",
                config_path="pretrained/config_v1.json",
                device=self.device
            )
        else:
            print("⚠️ Using TRAINABLE vocoder")
            self.vocoder = HiFiGANGenerator(self.config).to(self.device)
        
        # Set to eval mode
        self.generator.eval()
        self.ppg_extractor.eval()
        self.speaker_encoder.eval()
        self.vocoder.eval()
    
    def _optimize_models(self):
        """Optimize models for inference (stable version)"""

        # --------------------------
        # FP16 (safe on CUDA)
        # --------------------------
        if self.config.use_half_precision and torch.cuda.is_available():
            print("Converting models to FP16...")
            self.generator = self.generator.half()
            self.ppg_extractor = self.ppg_extractor.half()
            self.speaker_encoder = self.speaker_encoder.half()
            # self.vocoder = self.vocoder.half()
            if not self.use_pretrained_vocoder:
             self.vocoder = self.vocoder.half()

        # --------------------------
        # ⚠️ DISABLE torch.compile
        # --------------------------
        # DO NOT compile speech models with dynamic audio length
        # It causes CUDA Graph overwrite errors
        print("Skipping torch.compile for stability.")

        # --------------------------
        # Quantization (CPU only)
        # --------------------------
        if self.config.use_quantization and not torch.cuda.is_available():
            print("Quantizing models for CPU inference...")
            self.generator = torch.quantization.quantize_dynamic(
                self.generator, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )

    def get_model_info(self):
        return {
            "device": str(self.device),
            "half_precision": self.config.use_half_precision,
            "quantized": self.config.use_quantization,
        }
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state dicts
        self.generator.load_state_dict(checkpoint['G_I2C'], strict=False)
        
        if 'PPG_extractor' in checkpoint:
            self.ppg_extractor.load_state_dict(checkpoint['PPG_extractor'], strict=False)
        
        if 'Speaker_encoder' in checkpoint:
            self.speaker_encoder.load_state_dict(checkpoint['Speaker_encoder'], strict=False)
        
        # if 'Vocoder' in checkpoint:
        #     self.vocoder.load_state_dict(checkpoint['Vocoder'], strict=False)
        if not self.use_pretrained_vocoder:
            if 'Vocoder' in checkpoint:
                self.vocoder.load_state_dict(checkpoint['Vocoder'], strict=False)
        
        print("Checkpoint loaded successfully")
    
    @torch.no_grad()
    def convert(self, mel_dysarthric: torch.Tensor) -> torch.Tensor:

        single_sample = mel_dysarthric.dim() == 2
        if single_sample:
            mel_dysarthric = mel_dysarthric.unsqueeze(0)

        print("Input mel shape:", mel_dysarthric.shape)

        mel_dysarthric = mel_dysarthric.to(self.device)

        # dtype consistency
        # if self.config.use_half_precision:
        #     mel_dysarthric = mel_dysarthric.half()
        # else:
        #     mel_dysarthric = mel_dysarthric.float()
        mel_dysarthric = mel_dysarthric.float()
        # ---------------------------
        # Feature extraction
        # ---------------------------
        ppg = self.ppg_extractor(mel_dysarthric)
        speaker_emb = self.speaker_encoder(mel_dysarthric)

        # ---------------------------
        # 🔥 GENERATOR (MISSING BEFORE)
        # ---------------------------
        mel_clear = self.generator(ppg, speaker_emb)

        # ---------------------------
        # 🔥 FIX: log-mel → linear mel
        # ---------------------------
        mel_clear = torch.clamp(mel_clear, min=-11.5, max=2.0)
        print("mel_clear range:", mel_clear.min().item(), mel_clear.max().item())

        # mel_clear = torch.pow(10.0, mel_clear)

        print("Generated mel shape:", mel_clear.shape)

        # ---------------------------
        # Ensure correct shape (B, 80, T)
        # ---------------------------
        if mel_clear.dim() == 3 and mel_clear.shape[1] != 80:
            mel_clear = mel_clear.transpose(1, 2)
            print("Transposed mel shape:", mel_clear.shape)

        # dtype for vocoder
        # if self.config.use_half_precision:
        #     mel_clear = mel_clear.float()
        # else:
        #     mel_clear = mel_clear.float()
        mel_clear = mel_clear.float()
        # ---------------------------
        # Vocoder
        # ---------------------------
        audio_clear = self.vocoder(mel_clear)

        print("Generated audio shape:", audio_clear.shape)

        if single_sample:
            audio_clear = audio_clear.squeeze(0)

        return audio_clear
    
    @torch.no_grad()
    def convert_streaming(self, mel_chunk: torch.Tensor, 
                         context: Optional[Dict] = None) -> tuple:
        """
        Convert audio in streaming mode with context
        Args:
            mel_chunk: (n_mels, T_chunk)
            context: Dictionary containing previous context
        Returns:
            audio_chunk: (1, T_audio)
            new_context: Updated context
        """
        if context is None:
            context = {
                'speaker_emb': None,
                'prev_ppg': None
            }
        
        mel_chunk = mel_chunk.unsqueeze(0).to(self.device)
        # if self.config.use_half_precision:
        #     mel_chunk = mel_chunk.half()
        mel_chunk = mel_chunk.float()
        
        # Extract or reuse speaker embedding (cached for sessionf)
        if context['speaker_emb'] is None:
            context['speaker_emb'] = self.speaker_encoder(mel_chunk)
        
        # Extract PPG
        ppg = self.ppg_extractor(mel_chunk)
        
        # Generate clear mel
        mel_clear = self.generator(ppg, context['speaker_emb'])
        
        # Convert to audio
        # audio_clear = self.vocoder(mel_clear)
        mel_clear = torch.clamp(mel_clear, min=-11.5, max=2.0)
        # mel_clear = torch.pow(10.0, mel_clear)
        mel_clear = mel_clear.float()
        audio_clear = self.vocoder(mel_clear)
        
        # Update context
        context['prev_ppg'] = ppg
        
        return audio_clear.squeeze(0), context
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        def count_params(model):
            return sum(p.numel() for p in model.parameters())
        
        return {
            'generator_params': count_params(self.generator),
            'ppg_extractor_params': count_params(self.ppg_extractor),
            'speaker_encoder_params': count_params(self.speaker_encoder),
            'vocoder_params': count_params(self.vocoder),
            'device': str(self.device),
            'half_precision': self.config.use_half_precision,
            'quantized': self.config.use_quantization
        }
