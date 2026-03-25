# backend/app/training/trainer.py
import os
import time
import traceback
from typing import Optional
import tempfile
import shutil
import signal
import sys
from backend.app.preprocessing.feature_extractor import FeatureExtractor


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from backend.app.utils.debug import DebugMonitor
from backend.app.models.pretrained_vocoder import PretrainedHiFiGAN

# -------------------------
# Compatibility helpers
# -------------------------
# Provide a single autocast_context(enabled) that works across torch versions.
try:
    # PyTorch >= 2.0: torch.amp.autocast supports device_type arg
    from torch.amp import autocast as _autocast_new

    def autocast_context(enabled: bool):
        # prefer device_type 'cuda' when CUDA is used; newer API accepts device_type argument
        # caller will ensure enabled only when CUDA and use_amp are True
        return _autocast_new(device_type="cuda", enabled=enabled)

except Exception:
    # Older PyTorch (or fallback): use torch.cuda.amp.autocast(enabled=...)
    try:
        from torch.cuda.amp import autocast as _autocast_old

        def autocast_context(enabled: bool):
            return _autocast_old(enabled=enabled)
    except Exception:
        # final fallback: no-op context manager
        from contextlib import nullcontext

        def autocast_context(enabled: bool):
            return nullcontext()


# -------------------------
# Trainer class
# -------------------------
class Trainer:
    """
    Production-ready trainer with:
      - mixed precision (torch.amp) on CUDA
      - optional torch.compile
      - dataloader tuning (persistent_workers/prefetch_factor)
      - safe vocoder offload and throttled vocoder updates
      - gradient accumulation
      - robust multiprocessing fallback (disable workers if system doesn't allow CUDA shared tensors)
      - ETA / throughput logging
      - robust resume/load/save support (AMP scaler, optimizers, mid-epoch resume)
    """

    def __init__(self, config):
        self.config = config

        self.debug_feature_extractor = FeatureExtractor(self.config)
        # self.debug = DebugMonitor(sample_rate=16000)
        self.debug = DebugMonitor(sample_rate=self.config.audio.sample_rate)
        # device selection (allow override via config)
        device_cfg = getattr(config, "device", None)
        if device_cfg is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            try:
                self.device = torch.device(device_cfg)
            except Exception:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training bookkeeping
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        self.accum_steps = int(getattr(config.training, "accum_steps", 1))

        self.vocoder_update_interval = 1   # ✅ FIXED

        self.offload_vocoder = True        # ✅ KEEP

        self.use_amp = bool(getattr(config.training, "use_mixed_precision", True))
        self.use_compile = bool(getattr(config.training, "use_torch_compile", False)) and hasattr(torch, "compile")

        # dataloader workers (allow env override)
        env_workers = os.getenv("NUM_WORKERS")
        if env_workers:
            try:
                self.num_workers = int(env_workers)
            except Exception:
                self.num_workers = int(getattr(config.training, "num_workers", 2))
        else:
            self.num_workers = int(getattr(config.training, "num_workers", 2))

        # Try to avoid CUDA sharing issues by defaulting to single-process on systems where it fails.
        # The calling script can override NUM_WORKERS env var to force multiple workers.
        if self.num_workers < 0:
            self.num_workers = 0

        # cudnn autotune helps for fixed-size inputs
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        print("Initializing trainer components...")
        # initialize components
        self._init_models()
        self._init_optimizers()
        self._init_losses()
        self._init_data_loaders()

        # TensorBoard and checkpoints
        os.makedirs(getattr(config.paths, "log_dir", "logs"), exist_ok=True)
        os.makedirs(getattr(config.paths, "checkpoint_dir", "checkpoints"), exist_ok=True)
        self.writer = SummaryWriter(getattr(config.paths, "log_dir", "logs"))

        # AMP scaler (only enabled on CUDA)
        try:
            # new API supports specifying device but keep enabled flag backwards compatible
            self.scaler = GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))
        except Exception:
            # fallback
            self.scaler = GradScaler(enabled=(self.use_amp and self.device.type == "cuda"))

        print(f"[TRAINER] device={self.device}, amp={self.use_amp}, vocoder_offload={self.offload_vocoder}, compile={self.use_compile}, num_workers={self.num_workers}")
                # -------------------------
        # Graceful shutdown handling
        # -------------------------
        signal.signal(signal.SIGTERM, self._handle_kill)
        signal.signal(signal.SIGINT, self._handle_kill)   # Ctrl+C

    # -------------------------
    # MODELS
    # -------------------------
    def _init_models(self):
        # import lazily to avoid heavy imports on module load
        from ..models.generator import Generator
        from ..models.discriminator import MultiScaleDiscriminator
        from ..models.ppg_extractor import SimplePPGExtractor
        from ..models.speaker_encoder import SpeakerEncoder
        from ..models.vocoder import HiFiGANGenerator, HiFiGANDiscriminator

        print("Initializing models...")

        # generators / discriminators
        self.G_I2C = Generator(self.config).to(self.device)
        self.G_C2I = Generator(self.config).to(self.device)

        self.D_C = MultiScaleDiscriminator(self.config).to(self.device)
        self.D_I = MultiScaleDiscriminator(self.config).to(self.device)

        # frozen encoders (kept on device)
        self.PPG_extractor = SimplePPGExtractor(self.config).to(self.device)
        self.Speaker_encoder = SpeakerEncoder(self.config).to(self.device)
        for m in (self.PPG_extractor, self.Speaker_encoder):
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        # vocoder & its discriminator: allow offload to CPU to save GPU memory
        # self.Vocoder = HiFiGANGenerator(self.config)
        self.Vocoder = PretrainedHiFiGAN(
            checkpoint_path="pretrained/generator_v1",
            config_path="pretrained/config_v1.json",
            device=self.device
        )
        # self.D_V = HiFiGANDiscriminator()

        # if not self.offload_vocoder:
        #     self.Vocoder = self.Vocoder.to(self.device)
        #     self.D_V = self.D_V.to(self.device)
        # else:
        #     # ensure vocoder on CPU initially
        #     self.Vocoder = self.Vocoder.cpu()
        #     # self.D_V = self.D_V.cpu()

        # optional DataParallel wrapping (multi-GPU)
        if torch.cuda.device_count() > 1 and self.device.type == "cuda":
            self.G_I2C = nn.DataParallel(self.G_I2C)
            self.G_C2I = nn.DataParallel(self.G_C2I)
            self.D_C = nn.DataParallel(self.D_C)
            self.D_I = nn.DataParallel(self.D_I)

        # optionally compile models for speed (PyTorch 2.x)
        if self.use_compile:
            try:
                print("[TRAINER] Attempting torch.compile on generators and discriminators (if supported).")
                self.G_I2C = torch.compile(self.G_I2C)
                self.G_C2I = torch.compile(self.G_C2I)
            except Exception as e:
                print(f"[TRAINER] torch.compile failed or unsupported: {e}")
                self.use_compile = False

        self._print_model_sizes()

    def _print_model_sizes(self):
        def count_params(m):
            try:
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            except Exception:
                return -1

        try:
            print(f"Params - G_I2C: {count_params(self.G_I2C):,}, G_C2I: {count_params(self.G_C2I):,}")
            print(f"Params - D_C: {count_params(self.D_C):,}, D_I: {count_params(self.D_I):,}")
            # if not self.offload_vocoder:
            #     print(f"Params - Vocoder: {count_params(self.Vocoder):,}, D_V: {count_params(self.D_V):,}")
            # else:
            #     print("Vocoder kept on CPU to save GPU memory (offload_vocoder_to_cpu=True)")
            print(f"Params - Vocoder: {count_params(self.Vocoder):,}")
        except Exception:
            print("[TRAINER] Unable to print parameter counts (models may be compiled/wrapped).")

    # -------------------------
    # OPTIMIZERS & SCHEDULERS
    # -------------------------
    def _init_optimizers(self):
        lr = getattr(self.config.training, "learning_rate", 2e-4)
        betas = (getattr(self.config.training, "beta1", 0.5), getattr(self.config.training, "beta2", 0.999))

        self.optimizer_G = optim.AdamW(
            list(self.G_I2C.parameters()) + list(self.G_C2I.parameters()),
            lr=lr, betas=betas, weight_decay=0.01
        )

        self.optimizer_D = optim.AdamW(
            list(self.D_C.parameters()) + list(self.D_I.parameters()),
            lr=lr, betas=betas, weight_decay=0.01
        )

        # # vocoder optimizers
        # self.optimizer_V = optim.AdamW(self.Vocoder.parameters(), lr=lr, betas=betas, weight_decay=0.01)
        # self.optimizer_DV = optim.AdamW(self.D_V.parameters(), lr=lr, betas=betas, weight_decay=0.01)

        # schedulers
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(self.optimizer_G, gamma=0.999)
        self.scheduler_D = optim.lr_scheduler.ExponentialLR(self.optimizer_D, gamma=0.999)
        # self.scheduler_V = optim.lr_scheduler.ExponentialLR(self.optimizer_V, gamma=0.999)
        # self.scheduler_DV = optim.lr_scheduler.ExponentialLR(self.optimizer_DV, gamma=0.999)

    # -------------------------
    # LOSSES
    # -------------------------
    def _init_losses(self):
        from .losses import CombinedLoss
        # CombinedLoss expected to manage gan_loss, feature_matching_loss, mel_loss etc.
        self.criterion = CombinedLoss(self.config)

    # -------------------------
    # DATA LOADERS
    # -------------------------
    def _init_data_loaders(self):
        from .dataset import DysarthricSpeechDataset
        from backend.app.utils.collate import collate_fn

        print("Initializing datasets and DataLoaders...")
        pin_memory = bool(getattr(self.config.training, "pin_memory", True))

        # dataloader kwargs
        dl_kwargs = {
            "batch_size": getattr(self.config.training, "batch_size", 1),
            "shuffle": True,
            "num_workers": max(0, self.num_workers),
            "pin_memory": pin_memory,
            "drop_last": True,
            "collate_fn": collate_fn,
        }

        # if self.num_workers > 0:
        #     dl_kwargs["persistent_workers"] = True
        #     dl_kwargs["prefetch_factor"] = min(4, max(2, self.num_workers))
        if self.num_workers > 0 and self.device.type == "cuda":
            dl_kwargs["persistent_workers"] = True

        self.train_dataset = DysarthricSpeechDataset(self.config, split="train", cache=False)
        self.val_dataset = DysarthricSpeechDataset(self.config, split="val", cache=True)

        # If the environment cannot spawn subprocesses with CUDA shared tensors, user must pass num_workers=0
        try:
            self.train_loader = DataLoader(self.train_dataset, **dl_kwargs)
        except Exception as e:
            print(f"[WARN] DataLoader creation with num_workers={self.num_workers} failed: {e}. Falling back to num_workers=0.")
            dl_kwargs["num_workers"] = 0
            dl_kwargs.pop("persistent_workers", None)
            dl_kwargs.pop("prefetch_factor", None)
            self.train_loader = DataLoader(self.train_dataset, **dl_kwargs)
            self.num_workers = 0

        # validation loader: fewer workers
        val_workers = max(0, self.num_workers // 2)
        val_dl_kwargs = {
            "batch_size": getattr(self.config.training, "batch_size", 1),
            "shuffle": False,
            "num_workers": val_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
        }
        self.val_loader = DataLoader(self.val_dataset, **val_dl_kwargs)

        print(f"✅ DataLoaders ready ({self.num_workers} workers, prefetch={dl_kwargs.get('prefetch_factor','N/A')})")

    # -------------------------
    # CHECKPOINTS
    # -------------------------
    def save_checkpoint(self, is_best: bool = False, tag: str = "latest"):
        """
        Safe checkpoint saving with:
        - atomic write
        - rolling retention (latest, prev)
        """
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "G_I2C": self.G_I2C.state_dict(),
            "G_C2I": self.G_C2I.state_dict(),
            "D_C": self.D_C.state_dict(),
            "D_I": self.D_I.state_dict(),
            "Vocoder": self.Vocoder.state_dict() if hasattr(self.Vocoder, "state_dict") else None,
            # "D_V": self.D_V.state_dict() if hasattr(self.D_V, "state_dict") else None,
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            # "optimizer_V": self.optimizer_V.state_dict(),
            # "optimizer_DV": self.optimizer_DV.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
        }

        ckpt_dir = self.config.paths.checkpoint_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        final_path = os.path.join(ckpt_dir, f"checkpoint_{tag}.pt")
        prev_path = os.path.join(ckpt_dir, f"checkpoint_prev.pt")

        # 🔐 atomic write
        with tempfile.NamedTemporaryFile(dir=ckpt_dir, delete=False) as tmp:
            torch.save(ckpt, tmp.name)
            tmp.flush()
            os.fsync(tmp.fileno())

        # rotate
        if os.path.exists(final_path):
            shutil.move(final_path, prev_path)

        shutil.move(tmp.name, final_path)
        print(f"💾 Checkpoint saved: {final_path}")

        if is_best:
            best_path = os.path.join(ckpt_dir, "best_model.pt")
            shutil.copyfile(final_path, best_path)
            print(f"⭐ Best model updated (val loss={self.best_val_loss:.4f})")


    def load_checkpoint(self, checkpoint_path: str):
        """Resume full training state (models, optimizers, scaler, and counters)."""
        print(f"Loading checkpoint from {checkpoint_path} ...")
        # map_location ensures tensors go to the correct device
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Restore models (strict=False to be robust to compiled/wrapped models)
        try:
            self.G_I2C.load_state_dict(ckpt["G_I2C"], strict=False)
            self.G_C2I.load_state_dict(ckpt["G_C2I"], strict=False)
            self.D_C.load_state_dict(ckpt["D_C"], strict=False)
            self.D_I.load_state_dict(ckpt["D_I"], strict=False)
        except Exception as e:
            print(f"[WARN] Model state load raised: {e}. Attempting partial restore (strict=False).")

        # # Vocoder & D_V (load only if appropriate)
        # try:
        #     if ckpt.get("Vocoder") is not None and not self.offload_vocoder:
        #         self.Vocoder.load_state_dict(ckpt["Vocoder"], strict=False)
        #     if ckpt.get("D_V") is not None:
        #         self.D_V.load_state_dict(ckpt["D_V"], strict=False)
        # except Exception as e:
        #     print(f"[WARN] Vocoder/D_V load issue: {e}")

        # Restore optimizers safely (ignore dtype/device mismatches)
        try:
            self.optimizer_G.load_state_dict(ckpt["optimizer_G"])
            self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
            # self.optimizer_V.load_state_dict(ckpt["optimizer_V"])
            # self.optimizer_DV.load_state_dict(ckpt["optimizer_DV"])
        except Exception as e:
            print(f"[WARN] Optimizer state mismatch ignored: {e}")

        # Restore GradScaler (AMP)
        if "scaler" in ckpt and ckpt["scaler"] is not None and hasattr(self, "scaler"):
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                print("[WARN] Could not restore AMP scaler — continuing with fresh state.")

        # Restore counters
        saved_epoch = ckpt.get("epoch", 0)
        saved_global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))

        # compute resume behavior
        # if saved_global_step marks exact end of an epoch -> resume from next epoch
        # else resume same epoch but skip processed batches
        train_len = len(self.train_loader) if hasattr(self, "train_loader") else None
        if train_len is None or train_len == 0:
            # fallback: just set epoch to saved_epoch
            self.epoch = saved_epoch
            self.global_step = saved_global_step
            self._resume_skip_batches = 0
        else:
            # determine index within epoch
            start_batch = saved_global_step % train_len
            # if start_batch == 0 and saved_global_step != 0 then that means saved at epoch boundary
            if start_batch == 0 and saved_global_step != 0:
                # resume from next epoch
                self.epoch = saved_epoch + 1
                self.global_step = saved_global_step
                self._resume_skip_batches = 0
            else:
                # resume same epoch and skip start_batch items
                self.epoch = saved_epoch
                self.global_step = saved_global_step
                self._resume_skip_batches = start_batch

        print(f"✅ Loaded checkpoint from epoch {saved_epoch} (step {saved_global_step}). Resuming epoch={self.epoch}, skip_batches={getattr(self,'_resume_skip_batches',0)}")

        # -------------------------
    # SIGNAL HANDLER (safe shutdown)
    # -------------------------
    def _handle_kill(self, signum, frame):
        print(f"\n[SIGNAL] Received signal {signum}. Saving checkpoint before exit...")
        try:
            self.save_checkpoint()
        except Exception as e:
            print(f"[SIGNAL] Failed to save checkpoint: {e}")
        finally:
            sys.exit(0)



    # -------------------------
    # TRAIN EPOCH
    # -------------------------
    def train_epoch(self):
        # set modes
        self.G_I2C.train()
        self.G_C2I.train()
        self.D_C.train()
        self.D_I.train()
        self.PPG_extractor.eval()
        self.Speaker_encoder.eval()

        skipped_batches = 0
        total_batches = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}", unit="batch")

        gen_steps = 0
        epoch_start = time.time()

        # if resuming mid-epoch, skip first _resume_skip_batches batches
        resume_skip = getattr(self, "_resume_skip_batches", 0)
        if resume_skip:
            print(f"[RESUME] Skipping first {resume_skip} batches of epoch {self.epoch} (already processed).")
        for batch_idx, batch in enumerate(pbar):
            # skip resume batches
            if resume_skip and batch_idx < resume_skip:
                # update progress counters but do not train on this batch
                continue
            # Once we've skipped, clear flag
            if resume_skip and batch_idx >= resume_skip:
                # make sure we don't skip next epoch
                self._resume_skip_batches = 0
                resume_skip = 0

            if batch is None or not isinstance(batch, dict):
                skipped_batches += 1
                continue

            batch_start = time.time()
            try:
                # move mel tensors once with non_blocking (requires pin_memory True)
                mel_I = batch["dysarthric_mel"].to(self.device, non_blocking=True)
                mel_C = batch["clear_mel"].to(self.device, non_blocking=True)
                # 🔥 CRITICAL FIX: remove extra channel dimension
                if mel_I.dim() == 4:
                    mel_I = mel_I.squeeze(1)

                if mel_C.dim() == 4:
                    mel_C = mel_C.squeeze(1)
            except Exception as e:
                print(f"[Train] Skipping invalid batch {batch_idx}: {e}")
                skipped_batches += 1
                continue

            # precompute embeddings under no_grad (fast)
            with torch.no_grad():
                ppg_I = self.PPG_extractor(mel_I)
                ppg_C = self.PPG_extractor(mel_C)
                spk_I = self.Speaker_encoder(mel_I)
                spk_C = self.Speaker_encoder(mel_C)

            # ---------- Generator step ----------
            if self.global_step % self.accum_steps == 0:
              self.optimizer_G.zero_grad(set_to_none=True)
            with autocast_context(enabled=(self.use_amp and self.device.type == "cuda")):

                # -------------------------
                # Forward: I → C
                # -------------------------
                fake_C = self.G_I2C(ppg_I, spk_I)
                noise_std = getattr(self.config.training, "mel_noise_std", 0.0)

                if noise_std > 0:
                    fake_C = fake_C + noise_std * torch.randn_like(fake_C)
                fake_C = torch.clamp(fake_C, -11.5, 2.0)
                fake_C = torch.nan_to_num(fake_C, nan=0.0, posinf=2.0, neginf=-11.5)


                min_len_C = min(fake_C.size(2), mel_C.size(2))
                fake_C = fake_C[:, :, :min_len_C]
                mel_C_trim = mel_C[:, :, :min_len_C]

                ppg_fake_C = self.PPG_extractor(fake_C)
                rec_I = self.G_C2I(ppg_fake_C, spk_I)

                # -------------------------
                # Forward: C → I
                # -------------------------
                fake_I = self.G_C2I(ppg_C, spk_C)
                fake_I = torch.clamp(fake_I, -11.5, 2.0)
                fake_I = torch.nan_to_num(fake_I, nan=0.0, posinf=2.0, neginf=-11.5)

                min_len_I = min(fake_I.size(2), mel_I.size(2))
                fake_I = fake_I[:, :, :min_len_I]
                mel_I_trim = mel_I[:, :, :min_len_I]

                ppg_fake_I = self.PPG_extractor(fake_I)
                rec_C = self.G_I2C(ppg_fake_I, spk_C)
                # -------------------------
                # Discriminator predictions
                # -------------------------
                disc_fake_C, feat_fake_C = self.D_C(fake_C.float())
                disc_fake_I, feat_fake_I = self.D_I(fake_I.float())

                with torch.no_grad():
                    disc_real_C, feat_real_C = self.D_C(mel_C_trim.float())
                    disc_real_I, feat_real_I = self.D_I(mel_I_trim.float())

                # -------------------------
                # Vocoder (for perceptual loss)
                # -------------------------
                try:
                    mel_for_vocoder = torch.clamp(fake_C.detach(), -11.5, 2.0)

                    # if self.offload_vocoder:
                    #     # with torch.no_grad():
                    #     #     audio_fake_C = self.Vocoder(mel_for_vocoder.float().cpu())

                    #     # audio_fake_C = audio_fake_C.to(self.device, dtype=torch.float32, non_blocking=True)
                    #     with torch.no_grad():
                    #         audio_fake_C = audio_fake_C.to(self.device)
                    with torch.no_grad():
                        if self.offload_vocoder:
                            audio_fake_C = self.Vocoder(mel_for_vocoder.float().cpu()).to(self.device)
                        else:
                            audio_fake_C = self.Vocoder(mel_for_vocoder.float().to(self.device))

                    audio_real_C = batch["clear_audio"].unsqueeze(1).to(self.device)
                    # align length
                    min_len = min(audio_fake_C.size(-1), audio_real_C.size(-1))
                    audio_fake_C = audio_fake_C[..., :min_len]
                    audio_real_C = audio_real_C[..., :min_len]
                    # 🔥 CRITICAL FIX: stabilize waveform before loss
                    audio_fake_C = torch.clamp(audio_fake_C, -1.0, 1.0)
                    audio_real_C = torch.clamp(audio_real_C, -1.0, 1.0)

                    audio_fake_C = torch.nan_to_num(audio_fake_C, nan=0.0)
                    audio_real_C = torch.nan_to_num(audio_real_C, nan=0.0)

                except Exception as e_v:
                    print(f"[WARN] Vocoder forward failed: {e_v}")
                    sr = getattr(self.config, "audio", {}).get("sample_rate", 16000)
                    audio_fake_C = torch.zeros((mel_C.size(0), 1, sr), device=self.device)
                    audio_real_C = batch["clear_audio"].unsqueeze(1).to(self.device)
                    # align length
                    min_len = min(audio_fake_C.size(-1), audio_real_C.size(-1))
                    audio_fake_C = audio_fake_C[..., :min_len]
                    audio_real_C = audio_real_C[..., :min_len]
                    # 🔥 CRITICAL FIX: stabilize waveform before loss
                    audio_fake_C = torch.clamp(audio_fake_C, -1.0, 1.0)
                    audio_real_C = torch.clamp(audio_real_C, -1.0, 1.0)

                    audio_fake_C = torch.nan_to_num(audio_fake_C, nan=0.0)
                    audio_real_C = torch.nan_to_num(audio_real_C, nan=0.0)

                # -------------------------
                # Speaker features
                # -------------------------
                spk_fake_C = self.Speaker_encoder(fake_C)
                spk_fake_I = self.Speaker_encoder(fake_I)

                # -------------------------
                # Generator loss
                # -------------------------
                loss_G, losses_G = self.criterion.compute_generator_loss(
                    disc_fake_C + disc_fake_I,
                    feat_real_C + feat_real_I,
                    feat_fake_C + feat_fake_I,
                    fake_C,
                    mel_C_trim,  # ✅ FIXED
                    audio_fake_C,
                    audio_real_C,
                    ppg_fake_C,
                    ppg_I,
                    spk_fake_C,
                    spk_I,
                    rec_I,
                    mel_I_trim,  # ✅ FIXED
                )

                loss_G = loss_G / max(1, self.accum_steps)
            # =========================
            # 🔥 Generator Backward + Step (CRITICAL)
            # =========================
            if self.use_amp and self.device.type == "cuda":
                try:
                    self.scaler.scale(loss_G).backward()
                except Exception:
                    loss_G.backward()
            else:
                loss_G.backward()

            # Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(
                list(self.G_I2C.parameters()) + list(self.G_C2I.parameters()),
                max_norm=10.0
            )

            # Optimizer step (with accumulation)
            if (self.global_step + 1) % self.accum_steps == 0:
                if self.use_amp and self.device.type == "cuda":
                    try:
                        self.scaler.step(self.optimizer_G)
                    except Exception:
                        self.optimizer_G.step()
                else:
                    self.optimizer_G.step()
            # AFTER loss computation
            if self.global_step % 200 == 0:
                self.debug.check_nan(fake_C, "fake_C")
                self.debug.check_nan(mel_C_trim, "mel_C")
                self.debug.log_mel_stats(fake_C, mel_C_trim, self.global_step)
                # self.debug.save_audio(self.Vocoder, fake_C, mel_C_trim, self.global_step)
                # fake_C_debug = torch.clamp(fake_C.detach(), -11.5, 2.0)
                # real_C_debug = torch.clamp(mel_C_trim.detach(), -11.5, 2.0)

                # self.debug.save_audio(self.Vocoder, fake_C_debug, real_C_debug, self.global_step)
                # ✅ use ORIGINAL mel (NOT trimmed)
                # real_C_debug = torch.clamp(mel_C.detach(), -11.5, 2.0)
                # 🔥 RECOMPUTE MEL from raw audio (MATCH test_vocoder)
                with torch.no_grad():
                    audio_real = batch["clear_audio"][0].cpu()

                    mel_real = self.debug_feature_extractor.extract_mel(audio_real.unsqueeze(0))
                    print("DEBUG MEL RANGE:", mel_real.min().item(), mel_real.max().item())

                    real_C_debug = torch.clamp(mel_real, -11.5, 2.0).to(self.device)

                # fake can stay trimmed (generator output)
                fake_C_debug = torch.clamp(fake_C.detach(), -11.5, 2.0)

                self.debug.save_audio(self.Vocoder, fake_C_debug, real_C_debug, self.global_step)
                self.debug.check_loss(loss_G, "Generator Loss")

            if self.global_step % 200 == 0:
                print(f"Step {self.global_step} | Loss_G: {loss_G.item():.4f}")
                print("Loss breakdown:",
                    losses_G["mel"].item(),
                    losses_G["wave"].item(),
                    losses_G["stft"].item())
            
            # ---------- Discriminator step ----------
            if self.global_step % self.accum_steps == 0:
                self.optimizer_D.zero_grad(set_to_none=True)
            try:
                disc_real_C, _ = self.D_C(mel_C_trim.float())
                disc_real_I, _ = self.D_I(mel_I_trim.float())
                disc_fake_C_det, _ = self.D_C(fake_C.detach().float())
                disc_fake_I_det, _ = self.D_I(fake_I.detach().float())

                loss_D = self.criterion.compute_discriminator_loss(
                    disc_real_C + disc_real_I, disc_fake_C_det + disc_fake_I_det
                )
                loss_D = loss_D / max(1, self.accum_steps)

                if self.use_amp and self.device.type == "cuda":
                    try:
                        self.scaler.scale(loss_D).backward()
                    except Exception as e:
                        print(f"[WARN] scaler.backward failed for loss_D: {e}; falling back to loss_D.backward()")
                        loss_D.backward()
                else:
                    loss_D.backward()

                torch.nn.utils.clip_grad_norm_(list(self.D_C.parameters()) + list(self.D_I.parameters()), max_norm=10.0)

                if (self.global_step + 1) % self.accum_steps == 0:
                    if self.use_amp and self.device.type == "cuda":
                        try:
                            self.scaler.step(self.optimizer_D)
                        except Exception as e:
                            print(f"[WARN] scaler.step failed for optimizer_D: {e}; falling back to optimizer_D.step()")
                            self.optimizer_D.step()
                    else:
                        self.optimizer_D.step()
                if (self.global_step + 1) % self.accum_steps == 0:
                    self.scaler.update()
            except RuntimeError as e:
                print(f"[Train] Discriminator step skipped at batch {batch_idx}: {e}")
                skipped_batches += 1
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()

            # # ---------- Vocoder (throttled & safe) ----------
            # if gen_steps % self.vocoder_update_interval == 0:
            #     self.optimizer_V.zero_grad(set_to_none=True)
            #     self.optimizer_DV.zero_grad(set_to_none=True)
            #     try:
            #         # detach mel inputs (prevent graph back to generator), force float32
            #         # 🔥 FIX: convert log-mel → linear mel
            #         mel_input = torch.clamp(mel_C.detach(), -11.5, 2.0)

            #         if self.offload_vocoder:
            #             # Run vocoder & D_V on CPU with normal FP32 training
            #             mel_input_cpu = mel_input.to("cpu", non_blocking=True)
            #             # Ensure models on CPU
            #             self.Vocoder = self.Vocoder.cpu()
            #             self.D_V = self.D_V.cpu()

            #             # CPU forward (we want grads for vocoder)
            #             audio_gen_cpu = self.Vocoder(mel_input_cpu)
            #             audio_real_cpu = (
            #                 batch.get("clear_audio", torch.zeros_like(audio_gen_cpu))
            #                 .unsqueeze(1)
            #                 .to("cpu", dtype=torch.float32, non_blocking=True)
            #             )

            #             min_len = min(audio_gen_cpu.size(-1), audio_real_cpu.size(-1))
            #             audio_gen_cpu = audio_gen_cpu[..., :min_len]
            #             audio_real_cpu = audio_real_cpu[..., :min_len]

            #             disc_real_v, feat_real_v = self.D_V(audio_real_cpu)
            #             disc_fake_v, feat_fake_v = self.D_V(audio_gen_cpu.detach())

            #             loss_V_adv = self.criterion.gan_loss.generator_loss(disc_fake_v)
            #             loss_V_fm = self.criterion.feature_matching_loss.forward(feat_real_v, feat_fake_v)
            #             # 🔥 FIX: waveform loss instead of mel loss
            #             loss_V_mel = torch.nn.functional.l1_loss(audio_gen_cpu, audio_real_cpu)

            #             # loss_V = loss_V_adv + 2.0 * loss_V_fm + 45.0 * loss_V_mel
            #             loss_V = loss_V_adv + 2.0 * loss_V_fm + 45.0 * loss_V_mel 
            #             loss_V = loss_V / max(1, self.accum_steps)

            #             # CPU backward/step (GradScaler isn't used on CPU)
            #             loss_V.backward()
            #             torch.nn.utils.clip_grad_norm_(self.Vocoder.parameters(), max_norm=10.0)
            #             if (self.global_step + 1) % self.accum_steps == 0:
            #                 self.optimizer_V.step()

            #             # Vocoder discriminator (CPU)
            #             disc_real_v, feat_real_v = self.D_V(audio_real_cpu.detach())
            #             disc_fake_v, feat_fake_v = self.D_V(audio_gen_cpu.detach())
            #             loss_DV = self.criterion.compute_discriminator_loss(disc_real_v, disc_fake_v)
            #             loss_DV = loss_DV / max(1, self.accum_steps)
            #             loss_DV.backward()
            #             torch.nn.utils.clip_grad_norm_(self.D_V.parameters(), max_norm=10.0)
            #             if (self.global_step + 1) % self.accum_steps == 0:
            #                 self.optimizer_DV.step()

            #         else:
            #             # GPU path: use AMP & GradScaler
            #             mel_input_gpu = mel_input.to(self.device, dtype=torch.float32, non_blocking=True)
            #             with autocast_context(enabled=(self.use_amp and self.device.type == "cuda")):
            #                 audio_gen = self.Vocoder(mel_input_gpu)
            #                 audio_real = (
            #                     batch.get("clear_audio", torch.zeros_like(audio_gen))
            #                     .unsqueeze(1)
            #                     .to(self.device, dtype=torch.float32, non_blocking=True)
            #                 )

            #             min_len = min(audio_gen.size(-1), audio_real.size(-1))
            #             audio_gen = audio_gen[..., :min_len]
            #             audio_real = audio_real[..., :min_len]

            #             disc_real_v, feat_real_v = self.D_V(audio_real)
            #             disc_fake_v, feat_fake_v = self.D_V(audio_gen.detach())

            #             loss_V_adv = self.criterion.gan_loss.generator_loss(disc_fake_v)
            #             loss_V_fm = self.criterion.feature_matching_loss.forward(feat_real_v, feat_fake_v)
            #             loss_V_mel = torch.nn.functional.l1_loss(audio_gen, audio_real)
            #             loss_V = loss_V_adv + 2.0 * loss_V_fm + 45.0 * loss_V_mel
            #             loss_V = loss_V / max(1, self.accum_steps)

            #             if self.use_amp and self.device.type == "cuda":
            #                 try:
            #                     self.scaler.scale(loss_V).backward()
            #                 except Exception as e:
            #                     print(f"[WARN] scaler.scale(loss_V).backward() failed: {e}; using loss_V.backward()")
            #                     loss_V.backward()
            #             else:
            #                 loss_V.backward()

            #             torch.nn.utils.clip_grad_norm_(self.Vocoder.parameters(), max_norm=10.0)
            #             if (self.global_step + 1) % self.accum_steps == 0:
            #                 if self.use_amp and self.device.type == "cuda":
            #                     try:
            #                         self.scaler.step(self.optimizer_V)
            #                         self.scaler.update()
            #                     except Exception as e:
            #                         print(f"[WARN] scaler.step failed for optimizer_V: {e}; falling back to optimizer_V.step()")
            #                         self.optimizer_V.step()
            #                 else:
            #                     self.optimizer_V.step()

            #             # vocoder discriminator (GPU)
            #             disc_real_v, feat_real_v = self.D_V(audio_real.detach())
            #             disc_fake_v, feat_fake_v = self.D_V(audio_gen.detach())
            #             loss_DV = self.criterion.compute_discriminator_loss(disc_real_v, disc_fake_v)
            #             loss_DV = loss_DV / max(1, self.accum_steps)
            #             if self.use_amp and self.device.type == "cuda":
            #                 try:
            #                     self.scaler.scale(loss_DV).backward()
            #                 except Exception as e:
            #                     print(f"[WARN] scaler.scale(loss_DV).backward() failed: {e}; using loss_DV.backward()")
            #                     loss_DV.backward()
            #             else:
            #                 loss_DV.backward()

            #             torch.nn.utils.clip_grad_norm_(self.D_V.parameters(), max_norm=10.0)
            #             if (self.global_step + 1) % self.accum_steps == 0:
            #                 if self.use_amp and self.device.type == "cuda":
            #                     try:
            #                         self.scaler.step(self.optimizer_DV)
            #                         self.scaler.update()
            #                     except Exception as e:
            #                         print(f"[WARN] scaler.step failed for optimizer_DV: {e}; falling back to optimizer_DV.step()")
            #                         self.optimizer_DV.step()
            #                 else:
            #                     self.optimizer_DV.step()

            #     except RuntimeError as e:
            #         print(f"[Train] Vocoder step skipped at batch {batch_idx}: {e}")
            #         skipped_batches += 1
            #         if "out of memory" in str(e).lower():
            #             torch.cuda.empty_cache()
            #     except Exception as ex:
            #         print(f"[Train] Vocoder unexpected error at batch {batch_idx}: {ex}")
            #         traceback.print_exc()
            #         skipped_batches += 1
            #         torch.cuda.empty_cache()

            # ---------- Logging + cleanup ----------
            if self.global_step % getattr(self.config.training, "log_every", 100) == 0:
                try:
                    self.writer.add_scalar("Train/Loss_G", losses_G.get("total", torch.tensor(0.0)).item(), self.global_step)
                except Exception:
                    pass

            # memory cleanup
            try:
                del loss_G, disc_fake_C, disc_fake_I, feat_fake_C, feat_fake_I
            except Exception:
                pass
            if "loss_D" in locals():
                del loss_D
            # torch.cuda.empty_cache()

            # step/time bookkeeping
            self.global_step += 1
           # --- autosave every N steps ---
            autosave_steps = 500  # or 1000 (configurable)

            if self.global_step % autosave_steps == 0:
                print(f"[AUTO-SAVE] Saving checkpoint at step {self.global_step}")
                self.save_checkpoint()

            batch_time = time.time() - batch_start
            steps_done = batch_idx + 1
            elapsed = time.time() - epoch_start
            avg_per_batch = elapsed / max(1, steps_done)
            eta = avg_per_batch * max(0, (total_batches - steps_done))
            pbar.set_postfix({"step": self.global_step, "batch_time": f"{batch_time:.2f}s", "ETA": f"{eta/3600:.2f}h"})
            gen_steps += 1

        print(f"✅ Epoch {self.epoch + 1} complete. Skipped {skipped_batches} invalid batches.")

        # step schedulers once per epoch
        try:
            self.scheduler_G.step()
            self.scheduler_D.step()
            # self.scheduler_V.step()
            # self.scheduler_DV.step()
        except Exception:
            pass

    # -------------------------
    # VALIDATION
    # -------------------------
    def validate(self):
        self.G_I2C.eval()
        if not self.offload_vocoder:
            self.Vocoder.eval()

        skipped_batches = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                if batch is None or not isinstance(batch, dict):
                    skipped_batches += 1
                    continue

                try:
                    # ✅ LOAD FIRST
                    mel_I = batch["dysarthric_mel"].to(self.device)
                    mel_C = batch["clear_mel"].to(self.device)

                    # 🔥 FIX: remove extra channel dim if present
                    if mel_I.dim() == 4:
                        mel_I = mel_I.squeeze(1)

                    if mel_C.dim() == 4:
                        mel_C = mel_C.squeeze(1)

                    # Optional debug
                    if batch_idx == 0:
                        print("mel_I:", mel_I.shape, "mel_C:", mel_C.shape)

                except Exception as e:
                    print(f"[Validation] Skipping batch {batch_idx}: {e}")
                    skipped_batches += 1
                    continue

                # Forward
                ppg_I = self.PPG_extractor(mel_I)
                spk_I = self.Speaker_encoder(mel_I)
                fake_C = self.G_I2C(ppg_I, spk_I)

                try:
                    # ✅ Align lengths
                    min_len = min(fake_C.size(2), mel_C.size(2))
                    if min_len == 0:
                        skipped_batches += 1
                        continue

                    fake_C_trim = fake_C[:, :, :min_len]
                    mel_C_trim = mel_C[:, :, :min_len]

                    loss = torch.nn.functional.l1_loss(fake_C_trim, mel_C_trim)
                    total_loss += loss.item()

                except Exception as e:
                    print(f"Skipped batch due to error: {e}")
                    skipped_batches += 1
                    continue

        avg_loss = total_loss / max(1, (len(self.val_loader) - skipped_batches))
        print(f"✅ Validation done. Skipped {skipped_batches} invalid batches.")

        try:
            self.writer.add_scalar("Val/Loss", avg_loss, self.epoch)
        except Exception:
            pass

        return avg_loss
    # -------------------------
    # HOIST TRAIN
    # -------------------------
    def train(self):
        num_epochs = int(getattr(self.config.training, "num_epochs", 10))
        save_every = int(getattr(self.config.training, "save_every", 5))

        print(f"🚀 Starting training for {num_epochs} epochs...")
        for e in range(self.epoch, num_epochs):
            self.epoch = e
            print(f"\n=== Epoch {e + 1}/{num_epochs} ===")
            self.train_epoch()

            if (e + 1) % save_every == 0:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True, tag="latest")
                else:
                    self.save_checkpoint(tag="latest")


        print("🏁 Training finished.")
        try:
            self.writer.close()
        except Exception:
            pass
