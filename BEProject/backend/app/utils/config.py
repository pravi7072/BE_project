# backend/app/utils/config.py
import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import torch


# ============================================================
# 🎧 Audio Configuration
# ============================================================
@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    n_mfcc: int = 40
    fmin: int = 0
    fmax: int = 8000

    # Real-time streaming
    chunk_size: int = 4096
    overlap: int = 512
    max_buffer_size: int = 32000


# ============================================================
# 🧠 Model Configuration
# ============================================================
@dataclass
class ModelConfig:
    """Model architecture configuration"""
    ppg_dim: int = 39
    speaker_emb_dim: int = 256

    # Reduced default channels to fit smaller GPUs
    generator_channels: int = 128
    discriminator_channels: int = 32
    n_res_blocks: int = 6

    # Vocoder parameters
    vocoder_upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    vocoder_upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    vocoder_resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    vocoder_resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )


# ============================================================
# ⚙️ Training Configuration (Performance Tuned)
# ============================================================
@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 3
    num_epochs: int = 50
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999

    # 💡 Performance settings
    accum_steps: int = 2                   # Gradient accumulation
    vocoder_update_interval: int = 2      # Update vocoder every N steps
    offload_vocoder_to_cpu: bool = True   # Saves GPU memory
    use_mixed_precision: bool = True       # Use torch.amp where possible

    # 🧵 Dataloader performance
    num_workers: int = max(2, os.cpu_count() // 2)
    pin_memory: bool = True
    prefetch_factor: int = 4               # helps pipeline batches faster

    # 🚀 Optional CUDA & optimization flags
    cudnn_benchmark: bool = True           # Enables auto-tuning for faster convs
    cudnn_deterministic: bool = False      # Faster if you don’t need reproducibility
    torch_compile: bool = False            # Enable for PyTorch 2.x (small speedup)

    # # Loss weights
    # lambda_gan: float = 1.0
    # lambda_cycle: float = 10.0
    # lambda_ppg: float = 5.0
    # lambda_speaker: float = 2.0
    # lambda_mel: float = 45.0
    # lambda_feat_match: float = 2.0
    lambda_gan: float = 1.0
    lambda_feat_match: float = 10.0   # 🔥 CRITICAL FIX
    lambda_cycle: float = 5.0
    lambda_ppg: float = 1.0
    lambda_speaker: float = 1.0

    # Logging & checkpoints
    save_every: int = 5
    log_every: int = 100
    eval_every: int = 1000


# ============================================================
# 🌐 Server Configuration
# ============================================================
@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_connections: int = 100
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# ============================================================
# 📁 Paths
# ============================================================
@dataclass
class PathConfig:
    data_root: Path = Path("./data")
    dysarthric_dir: Path = Path("./data/raw/1")
    clear_dir: Path = Path("./data/raw/0")
    checkpoint_dir: Path = Path("./checkpoints")
    log_dir: Path = Path("./logs")
    pretrained_dir: Path = Path("./pretrained")
    cache_dir: Path = Path("./cache")

    def __post_init__(self):
        for path in [
            self.data_root,
            self.checkpoint_dir,
            self.log_dir,
            self.pretrained_dir,
            self.cache_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 🧩 Main Config Class
# ============================================================
class Config:
    def __init__(self):
        self.audio = AudioConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.server = ServerConfig()
        self.paths = PathConfig()

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count()

        # Quantization / precision toggles
        self.use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
        self.use_half_precision = os.getenv("USE_HALF_PRECISION", "true").lower() == "true"

        # Apply fast CUDA settings
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = self.training.cudnn_benchmark
            torch.backends.cudnn.deterministic = self.training.cudnn_deterministic

        # Load environment overrides
        self._load_from_env()

    def _load_from_env(self):
        """Allow environment overrides for config parameters."""
        def _get_env_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, default))
            except Exception:
                return default

        if os.getenv("DATA_ROOT"):
            self.paths.data_root = Path(os.getenv("DATA_ROOT"))

        self.training.batch_size = _get_env_int("BATCH_SIZE", self.training.batch_size)
        self.training.num_epochs = _get_env_int("NUM_EPOCHS", self.training.num_epochs)
        self.server.port = _get_env_int("SERVER_PORT", self.server.port)
        self.training.num_workers = _get_env_int("NUM_WORKERS", self.training.num_workers)

        if os.getenv("OFFLOAD_VOCODER"):
            self.training.offload_vocoder_to_cpu = os.getenv("OFFLOAD_VOCODER").lower() in ("1", "true", "yes")
        if os.getenv("USE_MIXED_PREC"):
            self.training.use_mixed_precision = os.getenv("USE_MIXED_PREC").lower() in ("1", "true", "yes")

        # Optional compile mode for PyTorch 2.x
        if os.getenv("TORCH_COMPILE", "false").lower() in ("1", "true", "yes"):
            self.training.torch_compile = True
