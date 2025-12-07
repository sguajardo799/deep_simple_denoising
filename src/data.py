import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F

from torchaudio.datasets import SPEECHCOMMANDS

import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import torch.nn.functional as F

class NoisySpeechCommands(Dataset):
    def __init__(
        self,
        root: str,
        download: bool = True,
        target_sample_rate: int = 16000,
        duration: float = 1.0,              # segundos
        snr_range=(0, 20),                  # SNR en dB [min, max]
        noise_type: str = "white",          # "white" u otros
        add_reverb: bool = False,
        reverb_prob: float = 0.5,
        rir_duration_range=(0.05, 0.4),
        rir_decay_range=(3.0, 7.0),
        max_items: int | None = None,       # NUEVO: limitar nº de muestras
    ):
        super().__init__()
        self.base = SPEECHCOMMANDS(root=root, download=download)
        self.target_sr = target_sample_rate
        self.num_samples = int(target_sample_rate * duration)
        self.snr_range = snr_range
        self.noise_type = noise_type

        self.add_reverb = add_reverb
        self.reverb_prob = reverb_prob
        self.rir_duration_range = rir_duration_range
        self.rir_decay_range = rir_decay_range

        # Índices usados (para poder recortar el dataset)
        all_indices = list(range(len(self.base)))
        if max_items is not None:
            all_indices = all_indices[:max_items]
        self.indices = all_indices

        # Resampler
        example_waveform, sr, *_ = self.base[0]
        self.need_resample = (sr != target_sample_rate)
        if self.need_resample:
            self.resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        else:
            self.resampler = None

    def __len__(self):
        return len(self.indices)

    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        T = wav.shape[-1]
        if T > self.num_samples:
            wav = wav[..., :self.num_samples]
        elif T < self.num_samples:
            pad = self.num_samples - T
            wav = F.pad(wav, (0, pad))
        return wav

    def _add_noise(self, clean: torch.Tensor) -> torch.Tensor:
        snr_db = torch.empty(1).uniform_(*self.snr_range).item()
        if self.noise_type == "white":
            noise = torch.randn_like(clean)
        else:
            noise = torch.randn_like(clean)

        sig_power = clean.pow(2).mean()
        noise_power = noise.pow(2).mean() + 1e-8
        target_noise_power = sig_power / (10 ** (snr_db / 10))
        noise = noise * torch.sqrt(target_noise_power / noise_power)
        noisy = clean + noise
        return noisy

    def _apply_reverb(self, clean: torch.Tensor) -> torch.Tensor:
        if not self.add_reverb:
            return clean
        if torch.rand(1).item() > self.reverb_prob:
            return clean

        sr = self.target_sr
        min_len = max(1, int(self.rir_duration_range[0] * sr))
        max_len = max(min_len + 1, int(self.rir_duration_range[1] * sr))
        rir_len = torch.randint(min_len, max_len + 1, (1,)).item()

        t = torch.arange(rir_len, dtype=clean.dtype, device=clean.device) / sr
        decay = torch.empty(1, device=clean.device).uniform_(
            *self.rir_decay_range
        ).item()

        rir = torch.randn(rir_len, dtype=clean.dtype, device=clean.device)
        rir = rir * torch.exp(-decay * t)
        rir = rir / (rir.abs().sum() + 1e-8)

        kernel = rir.view(1, 1, -1)
        x = clean.unsqueeze(0)  # (1,1,T)
        y = F.conv1d(x, kernel, padding=rir_len - 1)
        y = y.squeeze(0)
        return y

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        waveform, sr, label, speaker_id, utterance_number = self.base[base_idx]

        if self.need_resample:
            waveform = self.resampler(waveform)

        clean = self._fix_length(waveform)

        if self.add_reverb:
            clean = self._apply_reverb(clean)
            clean = self._fix_length(clean)

        noisy = self._add_noise(clean)

        return noisy, clean

def get_data_splits(dataset, split_path, val_ratio=0.2, seed=42):
    import torch
    import json
    import os
    from torch.utils.data import Subset

    if os.path.exists(split_path):
        print(f"Loading splits from {split_path}")
        with open(split_path, "r") as f:
            indices = json.load(f)
            train_indices = indices["train"]
            val_indices = indices["val"]
    else:
        print(f"Creating new splits and saving to {split_path}")
        n_total = len(dataset)
        n_val = int(val_ratio * n_total)
        n_train = n_total - n_val
        
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)
        
        train_indices = train_ds.indices
        val_indices = val_ds.indices
        
        with open(split_path, "w") as f:
            json.dump({"train": train_indices, "val": val_indices}, f)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
