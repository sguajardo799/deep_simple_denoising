import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torchaudio

from data import NoisySpeechCommands
from models import UNet2D  # <- ahora usamos la U-Net 2D


def main():
    # =========================
    # 1. Configuración general
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    root = "./data"   # cambia si quieres otra ruta

    batch_size = 8
    num_workers = 0

    max_epochs = 100
    learning_rate = 1e-3

    patience = 10          # early stopping: épocas sin mejora
    min_delta = 1e-4       # mejora mínima en val_loss para resetear paciencia
    log_interval = 100     # mostrar loss cada N steps durante el entrenamiento

    sample_rate = 16000
    n_fft = 512
    hop_length = 128
    n_mels = 80

    torch.manual_seed(42)

    # =========================
    # 2. Dataset y DataLoaders
    # =========================
    full_ds = NoisySpeechCommands(
        root=root,
        download=True,
        target_sample_rate=sample_rate,
        duration=10.0,
        snr_range=(0, 20),
        noise_type="white",
        add_reverb=True,
        reverb_prob=0.5,
        max_items=1000
    )

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # =========================
    # 3. Transforms de espectrograma
    # =========================
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        pad_mode="reflect",
        power=1.0,         # magnitud^2; puedes cambiar a 2.0 si quieres energía
    ).to(device)

    def waveform_to_logmel(wav: torch.Tensor) -> torch.Tensor:
        """
        wav: (B, 1, T)
        -> log-mel: (B, 1, n_mels, T_frames)
        """
        # quitar canal
        wav_mono = wav.squeeze(1)  # (B, T)
        mel = mel_transform(wav_mono)  # (B, n_mels, T_frames)
        log_mel = torch.log1p(mel)     # log(1 + mel) para estabilizar
        return log_mel.unsqueeze(1)    # (B, 1, n_mels, T_frames)

    # =========================
    # 4. Modelo, loss, optim
    # =========================
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        num_layers=4,
        kernel_size=3,
        use_batchnorm=True,
        dropout=0.0,
        final_activation=None,  # o "tanh" si quieres limitar la salida
    ).to(device)

    criterion = nn.MSELoss()          # loss en espacio log-mel
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # =========================
    # 5. Loop de entrenamiento
    # =========================
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):

        # ---- Fase de entrenamiento ----
        model.train()
        running_train_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

        for batch_idx, (noisy, clean) in enumerate(train_loop):
            noisy = noisy.to(device)   # [B,1,T]
            clean = clean.to(device)

            # Onda -> log-mel
            noisy_spec = waveform_to_logmel(noisy)   # [B,1,F,T']
            clean_spec = waveform_to_logmel(clean)   # [B,1,F,T']

            optimizer.zero_grad()
            pred_spec = model(noisy_spec)
            loss = criterion(pred_spec, clean_spec)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                train_loop.set_postfix(
                    loss=f"{running_train_loss / (batch_idx + 1):.4f}"
                )

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # ---- Fase de validación ----
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
            for noisy, clean in val_loop:
                noisy = noisy.to(device)
                clean = clean.to(device)

                noisy_spec = waveform_to_logmel(noisy)
                clean_spec = waveform_to_logmel(clean)

                pred_spec = model(noisy_spec)
                loss = criterion(pred_spec, clean_spec)
                running_val_loss += loss.item()

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # ---- Print resumen de época ----
        print(
            f"Epoch {epoch:03d}/{max_epochs} "
            f"| train_loss = {epoch_train_loss:.6f} "
            f"| val_loss = {epoch_val_loss:.6f}"
        )

        # ---- Early Stopping ----
        if epoch_val_loss + min_delta < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print("  -> Mejora en val_loss, guardando mejor modelo.")
        else:
            epochs_no_improve += 1
            print(f"  -> Sin mejora en val_loss ({epochs_no_improve}/{patience}).")

        if epochs_no_improve >= patience:
            print("Early stopping activado.")
            break

    # Restaurar mejor modelo
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f"Mejor val_loss alcanzado: {best_val_loss:.6f}")
    else:
        print("No se guardó ningún mejor estado (algo raro pasó).")

    # =========================
    # 6. Gráfico final Train vs Val
    # =========================
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolución Train vs Val Loss (log-mel)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
