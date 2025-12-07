import torch
import torchaudio
import matplotlib.pyplot as plt
from src.config import AudioConfig

def get_mel_transform(config: AudioConfig, device: str):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        center=True,
        pad_mode="reflect",
        power=1.0,
    ).to(device)

def waveform_to_logmel(wav: torch.Tensor, mel_transform: torch.nn.Module) -> torch.Tensor:
    """
    wav: (B, 1, T)
    -> log-mel: (B, 1, n_mels, T_frames)
    """
    wav_mono = wav.squeeze(1)        # (B, T)
    mel = mel_transform(wav_mono)    # (B, n_mels, T_frames)
    log_mel = torch.log1p(mel)       # log(1 + mel) para estabilizar
    return log_mel.unsqueeze(1)      # (B, 1, n_mels, T_frames)

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolución Train vs Val Loss (log-mel)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
