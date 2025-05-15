"""
This module provides a class to manage experiments with the PyTorch Lightning framework.
It is in charge of loading the configuration, setting up the data, model, and trainer,
and fitting the model.
"""

import glob
import json
import os

import nir
import snntorch

import lightning.pytorch as pl
import torch

from data.data_loaders import (
    HeraDataLoader,
    HeraDeltaNormLoader,
)
from data.data_module import ConfiguredDataModule
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import (
    LatencySpikeConverter,
)
from data.utils import reconstruct_patches
from evaluation import final_evaluation
from interfaces.data.raw_data_loader import RawDataLoader
from interfaces.data.spiking_data_module import SpikeConverter
from models.fc_latency import LitFcLatency


def data_source_from_config(config: dict) -> RawDataLoader:
    data_path = config.get("data_path")
    patch_size = config.get("patch_size")
    stride = config.get("stride")
    limit = config.get("limit")
    dataset = config.get("dataset")
    delta_normalization = config.get("delta_normalization")
    if dataset == "HERA":
        if delta_normalization:
            data_source = HeraDeltaNormLoader(
                data_path, patch_size=patch_size, stride=stride, limit=limit
            )
        else:
            data_source = HeraDataLoader(
                data_path, patch_size=patch_size, stride=stride, limit=limit
            )
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")
    return data_source


def dataset_from_config(
    config: dict, data_source: RawDataLoader, encoder: SpikeConverter
) -> ConfiguredDataModule:
    batch_size = config.get("batch_size")
    data_builder = DataModuleBuilder()
    data_builder.set_dataset(data_source)
    data_builder.set_encoding(encoder)
    dataset = data_builder.build(batch_size)
    return dataset


def model_from_config(config: dict) -> pl.LightningModule:
    model_type = config.get("type")
    beta = config.get("beta")
    alpha = config.get("alpha", beta)
    num_inputs = config.get("num_inputs")
    num_hidden = config.get("num_hidden")
    num_outputs = config.get("num_outputs")
    num_layers = config.get("num_layers", 2)
    num_hidden_layers = config.get("num_hidden_layers", num_layers)
    l1_weighting = config.get("l1_weighting", 0.0)
    l2_weighting = config.get("l2_weighting", 0.0)
    fan_in_weighting = config.get("fan_in_weighting", 0.0)
    max_connections_weighting = config.get("max_connections_weighting", 0.0)
    max_fan_in = config.get("max_fan_in", 63)
    max_connections = config.get("max_connections", 32000)
    if model_type == "FC_LATENCY" or model_type == "FC_LATENCY_XYLO" or model_type == "FC_LATENCY_FULL" or model_type == "FC_DIRECT_LATENCY":
        model = LitFcLatency(
            num_inputs, num_hidden, num_outputs, beta, alpha, num_layers, recurrent=False, l1_weighting=l1_weighting, l2_weighting=l2_weighting, fan_in_weighting=fan_in_weighting,
            max_connections=max_connections,
            max_fan_in=max_fan_in,
            max_connections_weighting=max_connections_weighting,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")
    return model


def trainer_from_config(config: dict, root_dir: str) -> pl.Trainer:
    num_gpus = torch.cuda.device_count()
    epochs = config.get("epochs")
    # patience = config.get("patience", 10)
    # early_stopping_callback = pl.callbacks.EarlyStopping(
    #     monitor="val_loss", mode="min", patience=patience, min_delta=1e-4
    # )
    if num_gpus > 0:
        trainer = pl.trainer.Trainer(
            max_epochs=epochs,
            benchmark=True,
            default_root_dir=root_dir,
            devices=num_gpus,
            num_nodes=config.get("num_nodes", 1),
        )
    else:
        trainer = pl.trainer.Trainer(
            max_epochs=epochs,
            benchmark=True,
            default_root_dir=root_dir,
            num_nodes=config.get("num_nodes", 1),
            accelerator="cpu",
            log_every_n_steps=4
        )
    return trainer


def encoder_from_config(config: dict) -> SpikeConverter:
    encoder = None
    if config.get("method") == "LATENCY":
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        encoder = LatencySpikeConverter(exposure=exposure, tau=tau, normalize=normalize)
    return encoder


class Experiment:
    def __init__(self, root_dir="./"):
        self.data_source = None
        self.dataset = None
        self.model = None
        self.trainer = None
        self.encoder = None
        self.configuration = None
        self.ready = False
        self.checkpoint_path = None
        self.root_dir = root_dir

    def from_config(self, config: dict):
        self.configuration = config
        if not self.encoder and self.configuration.get("encoder"):
            self.encoder = encoder_from_config(config.get("encoder"))
        if not self.data_source and self.configuration.get("data_source"):
            self.data_source = data_source_from_config(config.get("data_source"))
            if self.configuration.get("dataset") and self.encoder:
                self.dataset = dataset_from_config(
                    config.get("dataset"), self.data_source, self.encoder
                )
        if not self.model and self.configuration.get("model"):
            self.model = model_from_config(config.get("model"))
        if not self.trainer and self.configuration.get("trainer"):
            self.trainer = trainer_from_config(config.get("trainer"), self.root_dir)

    def from_checkpoint(self, working_dir: str):
        self.load_config(os.path.join(working_dir, "config.json"))
        checkpoint_dir = os.path.join(working_dir, "checkpoints", "*.ckpt")
        checkpoint_file = glob.glob(checkpoint_dir)[0]
        self.checkpoint_path = checkpoint_file
        self.from_config(self.configuration)

    def add_dataset(self, data_source: RawDataLoader):
        self.data_source = data_source
        self.dataset = None
        self.ready = False

    def save_config(self):
        out_dir = self.trainer.log_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as ofile:
            json.dump(self.configuration, ofile, indent=4)

    def save_model(self):
        out_dir = self.trainer.log_dir
        os.makedirs(out_dir, exist_ok=True)
        sample, _ = next(iter(self.dataset.train_dataloader()))
        nir_model = snntorch.export_to_nir(self.model, sample, "model.nir")
        nir.write(os.path.join(self.trainer.log_dir, "model.nir"), nir_model)

    def load_config(self, config_path: str):
        with open(config_path, "r") as ifile:
            self.configuration = json.load(ifile)

    def prepare(self):
        err_msg = ""
        if not self.ready:
            if self.data_source and self.encoder:
                if not self.dataset:
                    self.dataset = dataset_from_config(
                        self.configuration.get("dataset"),
                        self.data_source,
                        self.encoder,
                    )
            else:
                err_msg += "Data source not set.\n"
            if not self.model:
                err_msg += "Model not set.\n"
            if not self.trainer:
                err_msg += "Trainer not set.\n"
            else:
                if self.trainer.global_rank == 0:
                    self.save_config()
            if not self.encoder:
                err_msg += "Encoder not set.\n"
            else:
                self.model.set_converter(self.encoder)
            if err_msg != "":
                raise ValueError(err_msg)
            self.ready = True

    def train(self):
        if not self.ready:
            raise RuntimeError("Experiment not ready.")
        self.model.train()
        if self.checkpoint_path:
            self.trainer.fit(self.model, self.dataset, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.fit(self.model, self.dataset)

    def evaluate(self, plot=False):
        self.model.eval()
        metrics = self.trainer.test(self.model, self.dataset.test_dataloader())
        accuracy = metrics[0]["test_accuracy"]
        mse = metrics[0]["test_mse"]
        auroc = metrics[0]["test_auroc"]
        auprc = metrics[0]["test_auprc"]
        f1 = metrics[0]["test_f1"]
        output = json.dumps(
            {
                "accuracy": accuracy,
                "mse": mse,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
            }
        )
        # Write output
        with open(os.path.join(self.trainer.log_dir, "metrics.json"), "w") as ofile:
            json.dump(output, ofile, indent=4)
        if plot:
            try:
                mask_orig = reconstruct_patches(
                    self.data_source.fetch_test_y(),
                    self.data_source.original_size,
                    self.data_source.stride,
                )
                original_data = reconstruct_patches(
                    self.data_source.fetch_test_x(),
                    self.data_source.original_size,
                    self.data_source.stride,
                )
                final_evaluation(
                    self.model,
                    self.dataset,
                    self.encoder,
                    original_data,
                    mask_orig,
                    self.data_source.fetch_test_x(),
                    self.data_source.fetch_test_y(),
                    self.encoder.exposure,
                    self.trainer.log_dir,
                )
            except RuntimeError as e:
                print(f"Error during evaluation: {e}")
        return accuracy, mse, auroc, auprc, f1
