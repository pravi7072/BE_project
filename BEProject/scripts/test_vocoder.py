import torch
import torchaudio
from backend.app.models.pretrained_vocoder import PretrainedHiFiGAN
from backend.app.preprocessing.feature_extractor import FeatureExtractor
from backend.app.utils.config import Config
import soundfile as sf

config = Config()

# Load components
fe = FeatureExtractor(config)
vocoder = PretrainedHiFiGAN(
    checkpoint_path="pretrained/generator_v1",
    config_path="pretrained/config_v1.json",
    device="cuda"
)

# Load REAL audio file
audio, sr = torchaudio.load("data/raw/0/CF02_B1_C1_M2.wav")

# Resample if needed
if sr != config.audio.sample_rate:
    audio = torchaudio.functional.resample(audio, sr, config.audio.sample_rate)

# Extract mel (already LOG MEL)
mel = fe.extract_mel(audio)

# ❌ DO NOT CONVERT
# mel = torch.pow(10.0, mel)

# Move to GPU
mel = mel.to("cuda").float()

# DEBUG
print("mel range:", mel.min().item(), mel.max().item())

# Vocoder
with torch.no_grad():
    audio_out = vocoder(mel).cpu().squeeze().numpy()

sf.write("test_output.wav", audio_out, config.audio.sample_rate)

print("DONE")