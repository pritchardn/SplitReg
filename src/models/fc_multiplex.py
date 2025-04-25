"""
This module contains a distributed snnTorch based model (to complement hivemind)
"""
import torch
import lightning.pytorch as pl
from tqdm import tqdm

from evaluation import calculate_metrics


class LitFcMultiplex(pl.LightningModule):
    def __init__(
        self,
            models: list,
            input_bundles: list,
            output_bundles: list,
            converter,
    ):
        super().__init__()
        self.models = models
        self.input_bundles = input_bundles
        self.output_bundles = output_bundles
        self.converter = converter


    def _init_membranes(self, i):
        model = self.models[i]
        modules = list(model.modules())
        num_layers = (len(modules) - 3) // 2
        membranes = [lif.reset_mem() for lif in modules[2:num_layers * 2 + 1:2]]
        return membranes

    def calc_loss(self, y_hat, y):
        raise NotImplementedError("Model is for inference only.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Model is for inference only.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Model is for inference only.")

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
        raise NotImplementedError("Model is for inference only.")

    def _infer_slice(self, x, membranes, i):
        spike = None
        spike_counts = []
        model = self.models[i]
        modules = list(model.modules())[1:-2]
        num_layers = (len(list(model.modules())) - 3) // 2
        for n in range(num_layers):
            curr = modules[n * 2](x)
            spike, syn_mem, mem_mem = modules[n * 2 + 1](curr, membranes[n][0], membranes[n][1])
            membranes[n] = (syn_mem, mem_mem)
            x = spike
            spike_counts.append(torch.count_nonzero(spike).item())
        return spike, membranes[-1], spike_counts

    def _infer(self, x, i):
        full_spike = []
        spike_recordings = []
        # x -> [N x exp x C x freq x time]
        membranes = self._init_membranes(i)
        for t in range(x.shape[-1]):
            data = x[:, :, 0, :, t]
            spike, _, spike_rec = self._infer_slice(data, membranes, i)
            spike_recordings.append(spike_rec)
            full_spike.append(spike)
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        full_spike = torch.moveaxis(full_spike, 0, -1)
        full_spike = full_spike.unsqueeze(2)
        return full_spike

    def forward(self, x):
        output = torch.zeros_like(x)
        for i in range(len(self.models)):
            self.modes[i].to(x.device)
            outputs = self.output_bundles[i]
            inputs = self.input_bundles[i]
            output[..., outputs, :] = self._infer(x[..., inputs, :], i)
        return output.moveaxis(0, 1)
