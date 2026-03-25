# backend/app/utils/metrics.py
import torch
import numpy as np
from pesq import pesq
from pystoi import stoi
import librosa

class AudioMetrics:
    """Audio quality metrics"""
    
    @staticmethod
    def compute_pesq(reference, degraded, sr=16000):
        """Compute PESQ score"""
        try:
            score = pesq(sr, reference, degraded, 'wb')
            return score
        except Exception as e:
            print(f"Error computing PESQ: {e}")
            return None
    
    @staticmethod
    def compute_stoi(reference, degraded, sr=16000):
        """Compute STOI score"""
        try:
            score = stoi(reference, degraded, sr, extended=False)
            return score
        except Exception as e:
            print(f"Error computing STOI: {e}")
            return None
    
    @staticmethod
    def compute_mcd(mel_ref, mel_deg):
        """Compute Mel Cepstral Distortion"""
        mcd = torch.mean(torch.sqrt(torch.sum((mel_ref - mel_deg) ** 2, dim=1)))
        return mcd.item()
    
    @staticmethod
    def compute_snr(reference, degraded):
        """Compute Signal-to-Noise Ratio"""
        noise = reference - degraded
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    @staticmethod
    def compute_all_metrics(reference, converted, sr=16000):
        """Compute all metrics"""
        metrics = {}
        
        # PESQ
        pesq_score = AudioMetrics.compute_pesq(reference, converted, sr)
        if pesq_score is not None:
            metrics['pesq'] = pesq_score
        
        # STOI
        stoi_score = AudioMetrics.compute_stoi(reference, converted, sr)
        if stoi_score:
            metrics['stoi'] = stoi_score
        
        # SNR
        snr = AudioMetrics.compute_snr(reference, converted)
        metrics['snr'] = snr
        
        return metrics
