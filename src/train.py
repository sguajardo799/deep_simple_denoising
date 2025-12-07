import os
import csv
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from src.config import Config
from src.utils import waveform_to_logmel, plot_losses

def train_one_epoch(model, loader, criterion, optimizer, mel_transform, device, log_interval, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch_idx, (noisy, clean) in enumerate(loop):
        noisy = noisy.to(device)
        clean = clean.to(device)

        noisy_spec = waveform_to_logmel(noisy, mel_transform)
        clean_spec = waveform_to_logmel(clean, mel_transform)

        optimizer.zero_grad()
        pred_spec = model(noisy_spec)
        loss = criterion(pred_spec, clean_spec)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            loop.set_postfix(loss=f"{running_loss / (batch_idx + 1):.4f}")

    return running_loss / len(loader)

def validate_one_epoch(model, loader, criterion, mel_transform, device, epoch):
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
        for noisy, clean in loop:
            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_spec = waveform_to_logmel(noisy, mel_transform)
            clean_spec = waveform_to_logmel(clean, mel_transform)

            pred_spec = model(noisy_spec)
            loss = criterion(pred_spec, clean_spec)
            running_loss += loss.item()

    return running_loss / len(loader)

def train_model(config: Config, model, train_loader, val_loader, mel_transform):
    device = config.general.device
    results_dir = config.general.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Setup CSV logging
    csv_path = os.path.join(results_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_no_improve = 0

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, mel_transform, 
            device, config.training.log_interval, epoch
        )
        train_losses.append(train_loss)

        val_loss = validate_one_epoch(
            model, val_loader, criterion, mel_transform, device, epoch
        )
        val_losses.append(val_loss)

        # Log to CSV
        with open(csv_path, "a", newline="") as f:
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
    print(f"Métricas guardadas en: {csv_path}")
