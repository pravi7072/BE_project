# scripts/train.py
import torch.multiprocessing as mp
import sys
import os
import argparse
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# -----------------------------------------------------------------------------
# ✅ Environment setup for CUDA stability & memory management
# -----------------------------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: silence TensorFlow oneDNN noise

# Always use 'spawn' for torch multiprocessing (safer for CUDA)
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass # already set

# Add backend to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.utils.config import Config
from backend.app.training.trainer import Trainer


# -----------------------------------------------------------------------------
# ✅ Main training entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Dysarthric Speech Conversion Model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of DataLoader workers (set 0 to disable multiprocessing)")
    args = parser.parse_args()

    # Initialize config
    config = Config()

    # Override config from CLI args
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.device:
        config.device = torch.device(args.device)

    # 👇 New: control number of dataloader workers dynamically
    config.training.num_workers = args.num_workers

    print("=" * 80)
    print("Training Configuration:")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Data Root: {config.paths.data_root}")
    print(f"Num Workers: {config.training.num_workers}")
    print("=" * 80)

    # Initialize trainer
    trainer = Trainer(config)

    # Resume from checkpoint if provided
    # Resume priority:
    # 1. user-provided --resume
    # 2. checkpoint_latest.pt (auto)
    ckpt_dir = config.paths.checkpoint_dir
    auto_ckpt = os.path.join(ckpt_dir, "checkpoint_latest.pt")

    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif os.path.exists(auto_ckpt):
        print(f"🔄 Auto-resuming from {auto_ckpt}")
        trainer.load_checkpoint(auto_ckpt)


    # Train safely
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user.")
        trainer.save_checkpoint()
        print("✅ Checkpoint saved.")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint()
        print("✅ Checkpoint saved after exception.")


if __name__ == "__main__":
    main()
