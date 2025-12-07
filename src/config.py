import yaml
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GeneralConfig:
    device: str
    seed: int
    results_dir: str

@dataclass
class DataConfig:
    root: str
    download: bool
    duration: float
    snr_range: Tuple[float, float]
    noise_type: str
    add_reverb: bool
    reverb_prob: float
    max_items: int
    num_workers: int
    pin_memory: bool

@dataclass
class AudioConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int

@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    base_channels: int
    num_layers: int
    kernel_size: int
    use_batchnorm: bool
    dropout: float

@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    patience: int
    min_delta: float
    log_interval: int

@dataclass
class Config:
    general: GeneralConfig
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        
        return cls(
            general=GeneralConfig(**cfg_dict["general"]),
            data=DataConfig(**cfg_dict["data"]),
            audio=AudioConfig(**cfg_dict["audio"]),
            model=ModelConfig(**cfg_dict["model"]),
            training=TrainingConfig(**cfg_dict["training"]),
        )
