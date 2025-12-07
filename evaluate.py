import os
import torch
import torchaudio
import pandas as pd
from tqdm.auto import tqdm
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from src.config import Config
from src.data import NoisySpeechCommands, get_data_splits
from src.models import UNet2D
from src.features import get_transform
from src.utils import reconstruct_waveform

def evaluate():
    # 1. Load Config
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = Config.from_yaml(config_path)
    device = config.general.device
    print(f"Device: {device}")

    # 2. Setup Data (Validation Split)
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

    split_path = os.path.join(config.data.root, "splits.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found at {split_path}. Run training first.")
        
    _, val_ds = get_data_splits(full_ds, split_path, val_ratio=0.2, seed=config.general.seed)
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1, # Evaluate one by one for metrics
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # 3. Load Model
    model = UNet2D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        kernel_size=config.model.kernel_size,
        use_batchnorm=config.model.use_batchnorm,
        dropout=config.model.dropout,
        final_activation=None,
    ).to(device)

    best_model_path = os.path.join(config.general.results_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
         raise FileNotFoundError(f"Model file not found at {best_model_path}")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 4. Setup Metrics
    fs = config.audio.sample_rate
    stoi = ShortTimeObjectiveIntelligibility(fs, extended=False).to(device)
    pesq = PerceptualEvaluationSpeechQuality(fs, 'wb').to(device)
    si_sdr = ScaleInvariantSignalNoiseRatio().to(device)
    snr = SignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    # 5. Setup Transform
    transform = get_transform(config.audio, device)
    
    # Pre-calculate Spectrogram transform for phase extraction if needed
    # We need the complex spectrogram of the noisy signal to get the phase.
    # If transform_type is spectrogram, we can get it from torchaudio.transforms.Spectrogram(power=None)
    # If transform_type is melspectrogram, we still need STFT phase.
    
    stft = torchaudio.transforms.Spectrogram(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        power=None, # Complex
        center=True,
        pad_mode="reflect",
    ).to(device)

    results = []

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(tqdm(val_loader)):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Extract Phase from Noisy
            noisy_complex = stft(noisy.squeeze(1))
            noisy_phase = torch.angle(noisy_complex)

            # Forward Pass
            noisy_spec = transform(noisy.squeeze(1)).unsqueeze(1)
            pred_spec = model(noisy_spec)

            # Reconstruct Waveform
            pred_wav = reconstruct_waveform(pred_spec, noisy_phase, config.audio, device)
            
            # Ensure lengths match (inverse transform might have slight diff)
            min_len = min(pred_wav.shape[-1], clean.shape[-1])
            pred_wav = pred_wav[..., :min_len]
            clean = clean[..., :min_len]
            noisy = noisy[..., :min_len]

            # Calculate Metrics
            # Metrics expect (preds, target)
            m_stoi = stoi(pred_wav, clean)
            try:
                m_pesq = pesq(pred_wav, clean)
            except Exception as e:
                # PESQ can fail on silence or very short signals
                m_pesq = float('nan')
            
            m_sisdr = si_sdr(pred_wav, clean)
            m_snr = snr(pred_wav, clean)
            m_mse = mse(pred_wav, clean)
            m_mae = mae(pred_wav, clean)

            results.append({
                "id": i,
                "stoi": m_stoi.item(),
                "pesq": m_pesq.item() if not isinstance(m_pesq, float) else m_pesq,
                "si_sdr": m_sisdr.item(),
                "snr": m_snr.item(),
                "mse": m_mse.item(),
                "mae": m_mae.item()
            })

    # 6. Save Report
    df = pd.DataFrame(results)
    report_path = os.path.join(config.general.results_dir, "evaluation_report.csv")
    df.to_csv(report_path, index=False)
    
    print("\nEvaluation Summary:")
    print(df.mean(numeric_only=True))
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    evaluate()
