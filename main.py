import os
import torch
from torch.utils.data import DataLoader, random_split

from src.config import Config
from src.data import NoisySpeechCommands
from src.models import UNet2D
from src.utils import get_mel_transform
from src.train import train_model

def main():
    # 1. Cargar configuración
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = Config.from_yaml(config_path)
    print(f"Device: {config.general.device}")

    # 2. Setup Data
    # Asegurar directorios
    os.makedirs(config.data.root, exist_ok=True)
    
    full_ds = NoisySpeechCommands(
        root=config.data.root,
        download=config.data.download,
        target_sample_rate=config.audio.sample_rate,
        duration=config.data.duration,
        snr_range=config.data.snr_range,
        noise_type=config.data.noise_type,
        add_reverb=config.data.add_reverb,
        reverb_prob=config.data.reverb_prob,
        max_items=config.data.max_items,
    )

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # 3. Setup Model
    model = UNet2D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        kernel_size=config.model.kernel_size,
        use_batchnorm=config.model.use_batchnorm,
        dropout=config.model.dropout,
        final_activation=None,
    ).to(config.general.device)

    # 4. Setup Transform
    mel_transform = get_mel_transform(config.audio, config.general.device)

    # 5. Train
    train_model(config, model, train_loader, val_loader, mel_transform)

if __name__ == "__main__":
    main()
