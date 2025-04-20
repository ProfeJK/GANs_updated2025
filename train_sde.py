import os
import json
from datetime import datetime
import fire

import numpy as np
import torch
from torch.optim import Adam

from models.sde import Generator, DiscriminatorSimple, SdeGeneratorConfig
from training import WGANGPTrainer
from dataloaders import get_sde_dataloader
from utils.plotting import SDETrainingPlotter
from utils import get_accelerator_device

from dataclasses import dataclass
from models.layers import FFNNConfig
from evaluate_sde import plot_model_results


@dataclass
class AdamConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.9)
    weight_decay: float = 0.0

    def to_dict(self, prefix: str = ""):
        if prefix == "":
            return self.__dict__
        return {prefix + "_" + k: v for k, v in self.__dict__.items()}


def train_sdegan(params_file: str = None,
                 warm_start: bool = False,
                 epochs: int = 100,
                 device: str | None = None,
                 no_save: bool = False,
                 silent: bool = False,
                 hidden_size: int = 16) -> None:

    if params_file is not None:
        with open(params_file, 'r') as f:
            params = json.load(f)
    else:
        params = {
            "ISO": "ERCOT",
            "variables": ["TOTALLOAD", "WIND", "SOLAR"],
            "time_features": ["HOD"],
            "time_series_length": 24,
            "critic_iterations": 5,
            "penalty_weight": 10.0,
            "epochs": epochs,
            "total_epochs_trained": 0,
            "random_seed": 12345,
            "batch_size": 64
        }

        data_size = len(params["variables"])
        time_size = len(params["time_features"])
        initial_noise_size = 1  # ✅ Usamos solo el tamaño real del ruido

        gen_noise_embed_config = FFNNConfig(
            in_size=1,  # ✅ Cambiado de 2 a 1
            num_layers=2,
            num_units=32,
            out_size=hidden_size,
        )
        gen_drift_config = FFNNConfig(
            in_size=hidden_size + time_size,
            num_layers=3,
            num_units=64,
            out_size=hidden_size,
            final_activation="tanh"
        )
        gen
