import numpy as np
import nir
import torch
from rockpool.nn.modules import from_nir
import snntorch


def split_model(model: nir.NIRGraph, hw_config: dict, num_splits: int=2):
    out_layer_NN = hw_config["out_layer"]["NN"]
    in_layer_NN = hw_config["in_layer"]["NN"]
    # We don't care about f_in/out yet
    num_layers = (len(model.nodes) - 2) // 2
    # We'll also start by splitting a model in half.
    out_weights = [{} for _ in range(num_splits)]
    input_size = model.nodes["input"].input_type["input"]
    input_size[-2] //= num_splits
    output_size = model.nodes["output"].output_type["output"]
    output_size[-1] //= num_splits
    # Add inputs
    for out_weight in out_weights:
        out_weight["input"] = nir.Input({"input": input_size})
    for i in range(num_layers):
        weight_layer = model.nodes[f"layers_{i * 2}"]
        lif_layer = model.nodes[f"layers_{i * 2 + 1}"]
        n_out = weight_layer.weight.shape[0] // num_splits
        n_in = weight_layer.weight.shape[1] // num_splits
        for j, out_weight in enumerate(out_weights):
            out_weight[f"layers_{i * 2}"] = nir.Affine(
                weight_layer.weight[n_out * j:n_out * (j + 1), n_in * j:n_in * (j+1)],
                weight_layer.bias[n_out * j:n_out * (j + 1)],
                {'input': np.array([n_in])},
                {'output': np.array([n_out])},
                {}
            )
            out_weight[f"layers_{i * 2 + 1}"] = lif_layer
    # Append output layers
    for out_weight in out_weights:
        out_weight["output"] = nir.Output({"output": output_size})
    # Create new NIR models
    models = []
    for out_weight in out_weights:
        models.append(
            nir.NIRGraph(nodes=out_weight, edges=model.edges.copy(), input_type={'input': input_size}, output_type={'output': output_size}, metadata={})
        )
    return models


if __name__ == "__main__":
    # Load example model
    nir_model = nir.read("/Users/npritchard/PycharmProjects/SNN-RFI-SUPER/lightning_logs/version_69/model.nir")
    # nir_model = nir.read("C:\\Users\\Nicho\\PycharmProjects\\SNN-RFI-SUPER\\lightning_logs\\version_182\\model.nir")
    # Send into split
    models = split_model(nir_model, {"out_layer": {"NN": 4}, "in_layer": {"NN": 4}}, num_splits=4)
    # Load into SNNTorch
    snn_models = []
    for model in models:
        snn_models.append(snntorch.import_from_nir(model))
    # Load into Rockpool
    # rockpool_model_1 = from_nir(out_model_2)
    # rockpool_model_2 = from_nir(out_model_2)
    pass