import os
import csv
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.config import Config
from src.utils import plot_losses

import torchaudio
from src.losses import CompositeLoss

def train_one_epoch(model, loader, criterion, optimizer, transform, device, log_interval, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    scaler = torch.cuda.amp.GradScaler()
    
    # Pre-calculate Spectrogram transform for phase extraction if needed for CompositeLoss
    # We need the complex spectrogram of the noisy signal to get the phase.
    stft = torchaudio.transforms.Spectrogram(
        n_fft=512, # Assuming default, should ideally come from config but transform object doesn't expose it easily
        hop_length=128,
        power=None, # Complex
        center=True,
        pad_mode="reflect",
    ).to(device)

    for batch_idx, (noisy, clean) in enumerate(loop):
        noisy = noisy.to(device)
        clean = clean.to(device)

        noisy_spec = transform(noisy.squeeze(1)).unsqueeze(1)
        clean_spec = transform(clean.squeeze(1)).unsqueeze(1)
        
        # Extract phase for CompositeLoss
        if isinstance(criterion, CompositeLoss):
             with torch.no_grad():
                noisy_complex = stft(noisy.squeeze(1))
                noisy_phase = torch.angle(noisy_complex)
        else:
            noisy_phase = None

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            pred_spec = model(noisy_spec)
            
            if isinstance(criterion, CompositeLoss):
                loss = criterion(pred_spec, clean_spec, noisy_phase, clean)
            else:
                loss = criterion(pred_spec, clean_spec)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            loop.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, transform, device, epoch):
    model.eval()
    running_loss = 0.0
    
    # Pre-calculate Spectrogram transform for phase extraction
    stft = torchaudio.transforms.Spectrogram(
        n_fft=512,
        hop_length=128,
        power=None,
        center=True,
        pad_mode="reflect",
    ).to(device)
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
        for noisy, clean in loop:
            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_spec = transform(noisy.squeeze(1)).unsqueeze(1)
            clean_spec = transform(clean.squeeze(1)).unsqueeze(1)
            
            # Extract phase for CompositeLoss
            if isinstance(criterion, CompositeLoss):
                noisy_complex = stft(noisy.squeeze(1))
                noisy_phase = torch.angle(noisy_complex)
            else:
                noisy_phase = None

            pred_spec = model(noisy_spec)
            
            if isinstance(criterion, CompositeLoss):
                loss = criterion(pred_spec, clean_spec, noisy_phase, clean)
            else:
                loss = criterion(pred_spec, clean_spec)
                
            running_loss += loss.item()

    return running_loss / len(loader)

def train_model(config: Config, model, train_loader, val_loader, transform, criterion, optimizer):
    device = config.general.device
    results_dir = config.general.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Metrics logging setup
    metrics_path = os.path.join(config.general.results_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, transform, 
            device, config.training.log_interval, epoch
        )
        train_losses.append(train_loss)

        val_loss = validate_one_epoch(
            model, val_loader, criterion, transform, device, epoch
        )
        val_losses.append(val_loss)

        # Log to CSV
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        print(
            f"Epoch {epoch:03d}/{config.training.max_epochs} "
            f"| train_loss = {train_loss:.6f} "
            f"| val_loss = {val_loss:.6f}"
        )

        # Early Stopping
        if val_loss + config.training.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print("  -> Mejora en val_loss, guardando mejor modelo.")
        else:
            epochs_no_improve += 1
            print(f"  -> Sin mejora en val_loss ({epochs_no_improve}/{config.training.patience}).")

        if epochs_no_improve >= config.training.patience:
            print("Early stopping activado.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        print(f"Mejor val_loss alcanzado: {best_val_loss:.6f}")
        
        best_model_path = os.path.join(results_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        print(f"Modelo guardado en: {best_model_path}")
    else:
        print("No se guardó ningún mejor estado.")

    plot_losses(train_losses, val_losses, os.path.join(results_dir, "loss_curve.png"))
    print(f"Curva de loss guardada en: {os.path.join(results_dir, 'loss_curve.png')}")
    print(f"Métricas guardadas en: {metrics_path}")
