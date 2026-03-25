import os
import torch
import numpy as np
import soundfile as sf


class DebugMonitor:
    def __init__(self, sample_rate=16000, log_dir="debug"):
        self.sample_rate = sample_rate
        self.log_dir = log_dir

        os.makedirs(f"{log_dir}/audio", exist_ok=True)

    def check_nan(self, tensor, name="tensor"):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"❌ NaN/Inf detected in {name}")
            return True
        return False

    def log_mel_stats(self, fake_C, real_C, step):
        print(f"\n[DEBUG STEP {step}]")
        print(f"fake_C min/max: {fake_C.min().item():.4f}, {fake_C.max().item():.4f}")
        print(f"real_C min/max: {real_C.min().item():.4f}, {real_C.max().item():.4f}")

    # 
    def save_audio(self, vocoder, fake_C, real_C, step):
        try:
            with torch.no_grad():
                # 🔥 ALWAYS send mel to CPU (your vocoder supports this)
                mel_fake = torch.clamp(fake_C[0].float(), -11.5, 2.0).unsqueeze(0).cpu()
                mel_real = torch.clamp(real_C[0].float(), -11.5, 2.0).unsqueeze(0).cpu()

                audio_fake = vocoder(mel_fake).squeeze().cpu().numpy()
                audio_real = vocoder(mel_real).squeeze().cpu().numpy()

                audio_fake = np.nan_to_num(audio_fake)
                audio_real = np.nan_to_num(audio_real)

                audio_fake = np.clip(audio_fake, -1, 1)
                audio_real = np.clip(audio_real, -1, 1)

                sf.write(f"{self.log_dir}/audio/fake_{step}.wav", audio_fake, self.sample_rate)
                sf.write(f"{self.log_dir}/audio/real_{step}.wav", audio_real, self.sample_rate)

                print(f"🎧 Saved debug audio @ step {step}")

        except Exception as e:
            print(f"[DEBUG] Audio save failed: {e}")

    def check_loss(self, loss, name="loss"):
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ {name} is NaN/Inf → STOP TRAINING")
            return True
        return False