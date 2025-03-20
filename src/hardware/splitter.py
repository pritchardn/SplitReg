import numpy as np
import nir
import torch
from rockpool.nn.modules import from_nir
import snntorch


def split_model(model: nir.NIRGraph, hw_config: dict):
    out_layer_NN = hw_config["out_layer"]["NN"]
    in_layer_NN = hw_config["in_layer"]["NN"]
    # We don't care about f_in/out yet
    num_layers = (len(model.nodes) - 2) // 2
    # We'll also start by splitting a model in half
    out_weights_1 = {}
    out_weights_2 = {}
    input_size = model.nodes["input"].input_type["input"]
    input_size[-2] //= 2
    output_size = model.nodes["output"].output_type["output"]
    output_size[-1] //= 2
    # Add inputs
    out_weights_1["input"] = nir.Input({"input": input_size})
    out_weights_2["input"] = nir.Input({"input": input_size})
    for i in range(num_layers):
        weight_layer = model.nodes[f"layers_{i * 2}"]
        lif_layer = model.nodes[f"layers_{i * 2 + 1}"]
        n_out = weight_layer.weight.shape[0] // 2
        n_in = weight_layer.weight.shape[1] // 2
        out_weights_1[f"layers_{i * 2}"] = nir.Affine(
            weight_layer.weight[:n_out, :n_in],
            weight_layer.bias[:n_out],
            {'input': np.array([n_in])},
            {'output': np.array([n_out])},
            {}
        )
        out_weights_2[f"layers_{i * 2}"] = nir.Affine(
            weight_layer.weight[n_out:, n_in:],
            weight_layer.bias[n_out:],
            {'input': np.array([n_in])},
            {'output': np.array([n_out])},
            {}
        )
        out_weights_1[f"layers_{i * 2 + 1}"] = lif_layer
        out_weights_2[f"layers_{i * 2 + 1}"] = lif_layer
    # Append output layers
    out_weights_1["output"] = nir.Output({"output": output_size})
    out_weights_2["output"] = nir.Output({"output": output_size})
    # Create new NIR models
    out_model_1 = nir.NIRGraph(nodes=out_weights_1, edges=model.edges.copy(), input_type={'input': input_size}, output_type={'output': output_size}, metadata={})
    out_model_2 = nir.NIRGraph(nodes=out_weights_2, edges=model.edges.copy(), input_type={'input': input_size},
                               output_type={'output': output_size}, metadata={})
    return out_model_1, out_model_2



if __name__ == "__main__":
    # Load example model
    nir_model = nir.read("/Users/npritchard/PycharmProjects/SNN-RFI-SUPER/lightning_logs/version_69/model.nir")
    # Send into split
    out_model_1, out_model_2 = split_model(nir_model, {"out_layer": {"NN": 4}, "in_layer": {"NN": 4}})
    # Load into SNNTorch
    snntorch_model_1 = snntorch.import_from_nir(out_model_1)
    snntorch_model_2 = snntorch.import_from_nir(out_model_2)
    # Load into Rockpool
    # rockpool_model_1 = from_nir(out_model_2)
    # rockpool_model_2 = from_nir(out_model_2)
    pass