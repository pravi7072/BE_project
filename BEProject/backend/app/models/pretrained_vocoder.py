import torch
import torch.nn as nn
import json

from backend.app.models.hifigan_official import Generator
from hifigan.env import AttrDict

class PretrainedHiFiGAN(nn.Module):
    def __init__(self, checkpoint_path, config_path, device="cpu"):
        super().__init__()

        self.device = torch.device(device)

        with open(config_path) as f:
            config = AttrDict(json.load(f))

        self.model = Generator(config)
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        # 🔥 FIX
        if "generator" in state_dict:
            state_dict = state_dict["generator"]

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.model.remove_weight_norm()

    @torch.no_grad()
    def forward(self, mel):
        mel = mel.to(self.device).float()
        return self.model(mel)