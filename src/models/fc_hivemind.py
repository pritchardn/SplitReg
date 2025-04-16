"""
This module contains an implementation of a Pytorch lightning module capable of inference on entire spectrograms
by splitting the data across several independent xylo models.

This module is not intended for training, but for inference and as such, training functions will raise errors.
"""

import torch
import numpy as np
import lightning.pytorch as pl
from tqdm import tqdm
from evaluation import calculate_metrics



class LitFcHiveMind(pl.LightningModule):
    def __init__(self, xylo_models: list, input_bundles: list, output_bundles: list, converter):
        super().__init__()
        self.models = xylo_models
        self.input_bundles = input_bundles
        self.output_bundles = output_bundles
        self.converter = converter

    def _infer(self, x, i):
        new_x = x.detach().cpu().numpy().astype(np.int16)
        xylo_model = self.models[i]
        big_output = []
        for ex in new_x:
            output = []
            for t in range(x.shape[-1]):
                output.append(xylo_model(ex[..., t].squeeze(1))[0])
            big_output.append(np.stack(output, axis=-1))
        xylo_model.reset_state()
        big_output = np.stack(big_output, axis=0)
        # Calculate stats on batch
        # Decode inference
        big_output = np.expand_dims(big_output, axis=2)
        return torch.from_numpy(big_output).float()

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in tqdm(range(len(self.models)), desc="Model Iteration"):
            outputs = self.output_bundles[i]
            inputs = self.input_bundles[i]
            output[..., outputs, :] = self._infer(x[..., inputs, :], i)
        return output.moveaxis(0, 1)

    def calc_loss(self, y_hat, y):
        raise NotImplementedError("Module is for inference only")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Module is for inference only")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Module is for inference only")

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat = self(x)
        # Convert output to true output
        output_pred = self.converter.decode_inference(spike_hat.detach().cpu().numpy())
        accuracy, mse, auroc, auprc, f1 = calculate_metrics(
            y.detach().cpu().numpy(), output_pred
        )
        self.log("test_accuracy", accuracy, sync_dist=True)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_auroc", auroc, sync_dist=True)
        self.log("test_auprc", auprc, sync_dist=True)
        self.log("test_f1", f1, sync_dist=True)
        return accuracy, mse, auroc, auprc, f1

    def configure_optimizers(self):
        raise NotImplementedError("Module is for inference only")