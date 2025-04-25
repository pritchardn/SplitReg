import nir
import numpy as np
import snntorch
import warnings
from nir import NIRGraph
from rockpool.nn.modules import from_nir
import lightning.pytorch as pl
import json
import os

from data.utils import reconstruct_patches
from src.data.data_loaders import HeraDeltaNormLoader
from src.data.data_module_builder import DataModuleBuilder
from src.data.spike_converters import LatencySpikeConverter
from src.hardware.conversion_example import hardware_conversion, setup_xylo
from src.models.fc_hivemind import LitFcHiveMind
from src.models.fc_multiplex import LitFcMultiplex

alg_config = {
    "mode": "io_tied",
    "algorithm": "greedy"
}

xylo_hw_config = {
    "input": {
        "num_neurons": 8,
        "fan_out": 63
    },
    "output": {
        "num_neurons": 8,
        "fan_in": 63,
    },
    "hidden": {
        "fan_in": 63,
        "fan_out": 63,
        "num_neurons": 1000,
        "num_neurons_layer": -1,
        "num_layers": -1,
        "max_connections": 32000,
    }
}

def handle_split_arguments(hw_config: dict, split_config: dict):
    if split_config["mode"] != "io_tied":
        raise NotImplementedError(f"split mode {split_config['mode']} not implemented.")
    if split_config["algorithm"] != "greedy":
        raise NotImplementedError(f"split algorithm {split_config['algorithm']} not implemented.")
    if hw_config["input"]["num_neurons"] != hw_config["output"]["num_neurons"]:
        raise ValueError("input and output must have same number of neurons.")

def determine_output_bundles(num_outputs: int, hw_config: dict):
    max_output = hw_config["output"]["num_neurons"]
    groups = [list(range(i, min(i + max_output, num_outputs + 1))) for i in range(0, num_outputs, max_output)]
    return groups

def build_neuron_index(model: nir.NIRGraph, hw_config: dict, mode="maximal"):
    num_layers = (len(model.nodes) - 2) // 2
    output = {i:{} for i in range(num_layers)}
    fan_out = hw_config["hidden"]["fan_in"]
    for layer in range(num_layers-1, -1, -1):
        weight_layer = model.nodes[f"layers_{layer * 2}"]
        for i, weight in enumerate(weight_layer.weight[:]):
            curr_fan_out = min(fan_out, weight.shape[0])
            if mode == "maximal":
                # find best k weights
                best_inputs_indx = np.argpartition(np.abs(weight), -curr_fan_out)[-curr_fan_out:]
            elif mode == "random":
                best_inputs_indx = np.random.choice(weight.shape[0], curr_fan_out, replace=False)
            elif mode == "naive":
                bottom = max(0, i - curr_fan_out // 2)
                top = min(weight.shape[0], i + curr_fan_out // 2)
                if (top - i) < (i - bottom):  # Decrease bottom
                    bottom -= (curr_fan_out // 2) - (top - i)
                elif (top - i) > (i - bottom): # Increase top
                    diff = (curr_fan_out // 2) - (i - bottom)
                    top += diff
                # else - same so do nothing
                best_inputs_indx = np.arange(bottom, top)
            else:
                raise ValueError(f"unknown mode {mode}")
            best_inputs = weight[best_inputs_indx]
            assert len(best_inputs) <= curr_fan_out
            for neighbour, w in zip(best_inputs_indx, best_inputs):
                output[layer].setdefault(i, dict())[int(neighbour)] = w
    check_neuron_index_valid(output)
    return output

def find_maximal_valid_weights(neuron_index: dict, output_targets: list[int]):
    num_layers = len(neuron_index.keys())
    output = {i: {} for i in neuron_index.keys()}
    for layer in neuron_index.keys():
        target_neurons = neuron_index[layer].keys() if layer != num_layers-1 else output_targets
        for target in target_neurons:
            ancestors = neuron_index[layer][target]
            for ancestor, weight in ancestors.items():
                output[layer].setdefault(ancestor, 0.0)
                output[layer][ancestor] += np.abs(weight)
    return output

def num_hidden_neurons(current_weights: dict):
    num_neurons = 0
    for layer, neurons in current_weights.items():
        if layer != 0:
            num_neurons += len(neurons)
    return num_neurons


def num_hidden_connections(current_weights: dict, num_inputs: int, num_outputs: int):
    connection_count = 0
    num_layers = len(current_weights.keys())
    for i in range(1,num_layers):
        curr_layer_neurons = len(current_weights[i].keys())
        prev_layer_neurons = len(current_weights[i - 1].keys())
        connection_count += curr_layer_neurons * prev_layer_neurons
    return connection_count

def find_least_important_neuron(maximal_weights: dict):
    min_neuron = -1
    min_importance = float("inf")
    min_layer = -1
    for layer in range(len(maximal_weights.keys())):
        curr_neuron, curr_weight = find_least_important_neuron_layer(maximal_weights, layer)
        if curr_weight < min_importance:
            min_neuron = curr_neuron
            min_importance = curr_weight
            min_layer = layer
    return min_neuron, min_importance, min_layer

def find_random_neuron(maximal_weights: dict, min_input_width: int, layer: int=None):
    if layer is None:
        layer_sizes = [len(weights) for layer_idx, weights in maximal_weights.items()]
        choices = list(maximal_weights.keys())
        if layer_sizes[0] == min_input_width:
            layer_sizes.pop(0)
            choices.pop(0)
        total_size = sum(layer_sizes)
        probabilities = [size / total_size for size in layer_sizes]
        selected_layer = np.random.choice(choices, p=probabilities)
    else:
        selected_layer = layer
    selected_neuron = np.random.choice(list(maximal_weights[selected_layer].keys()))
    return selected_neuron, maximal_weights[selected_layer][selected_neuron], selected_layer

def find_naive_neuron(maximal_weights: dict, layer: int=None):
    if layer is None:
        layer_sizes = [len(weights) for layer_idx, weights in maximal_weights.items()]
        selected_layer = np.argmax(layer_sizes)
    else:
        selected_layer = layer
    selected_neuron = list(maximal_weights[selected_layer].keys())[-1]
    return selected_neuron, maximal_weights[selected_layer][selected_neuron], selected_layer

def find_least_important_neuron_layer(maximal_weights: dict, layer: int):
    min_neuron = -1
    min_importance = float("inf")
    for neuron, weight in maximal_weights[layer].items():
        weight /= len(maximal_weights[layer].keys())
        if weight < min_importance:
            min_importance = weight
            min_neuron = neuron
    return min_neuron, min_importance

def remove_target_backwards(target: int, maximal_weights: dict, neuron_index: dict, layer: int):
    maximal_weights[layer].pop(target)
    targets = [target]
    prev_layer = layer - 1
    curr_layer = layer
    while prev_layer >= 0:
        new_targets = []
        while targets:
            target = targets.pop(0)
            for weights in neuron_index[curr_layer].values():
                weights.pop(target, None)
            for neuron, weight in neuron_index[prev_layer][target].items():
                if neuron not in maximal_weights[prev_layer].keys():
                    continue
                maximal_weights[prev_layer][neuron] -= np.abs(weight)
                if np.abs(maximal_weights[prev_layer][neuron]) < np.finfo(float).eps:
                    new_targets.append(neuron)
                    maximal_weights[prev_layer].pop(neuron)
            if len(neuron_index[prev_layer][target]) == 0:
                del neuron_index[prev_layer][target]
        targets = new_targets
        curr_layer = prev_layer
        prev_layer -= 1
    return maximal_weights, neuron_index

def check_neuron_index_valid(neuron_index: dict):
    for layer, i in neuron_index.items():
        for j, weights in i.items():
            assert len(weights) <= 63

def remove_target_forwards(target: int, maximal_weights: dict, neuron_index: dict, layer: int):
    maximal_weights[layer].pop(target)
    targets = [target]
    next_layer = layer + 1
    curr_layer = layer
    while next_layer < len(maximal_weights[layer].keys()):
        new_targets = []
        while targets:
            target = targets.pop(0)
            descendents = set()
            for child, weights in neuron_index[curr_layer].items():
                weight = weights.pop(target, None)
                if weight is not None:
                    descendents.add((child, weight))
            for descendant, weight in descendents:
                if descendant not in maximal_weights[next_layer]:
                    continue
                maximal_weights[next_layer][descendant] -= np.abs(weight)
                if np.abs(maximal_weights[next_layer][descendant]) < np.finfo(float).eps:
                    new_targets.append(descendant)
                    maximal_weights[next_layer].pop(descendant)
            if len(neuron_index[curr_layer][target]) == 0:
                del neuron_index[curr_layer][target]
        targets = new_targets
        curr_layer = next_layer
        next_layer += 1
    return maximal_weights, neuron_index

def cull_weights_backwards(maximal_weights: dict, neuron_index: dict, max_neurons: int, min_input_width: int, mode: str):
    num_layers = len(neuron_index.keys())
    curr_neurons = num_hidden_neurons(maximal_weights)
    if curr_neurons < max_neurons:
        return maximal_weights, neuron_index
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_neurons > max_neurons:
        # Find target important neuron
        target = -1
        target_layer = -1
        if mode == "maximal":
            target, target_importance, target_layer = find_least_important_neuron(output)
        elif mode == "random":
            target, target_importance, target_layer = find_random_neuron(output, min_input_width)
        elif mode == "naive":
            target, target_importance, target_layer = find_naive_neuron(output)
        assert target != -1
        # Remove target
        output, new_neuron_index = remove_target_backwards(target, output, new_neuron_index, target_layer)
        # Cull recursively
        curr_neurons = num_hidden_neurons(output)
    return output, new_neuron_index

def cull_output_fan_in(maximal_weights: dict, neuron_index: dict, max_fan_in: int, min_input_width: int, mode: str):
    num_layers = len(neuron_index.keys())
    curr_neurons = num_hidden_neurons({1: maximal_weights[num_layers - 1]})
    if curr_neurons < max_fan_in:
        return maximal_weights, neuron_index
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_neurons > max_fan_in:
        # Find target important neuron
        target = -1
        target_layer = num_layers - 1
        if mode == "maximal":
            target, target_importance = find_least_important_neuron_layer(output, layer=target_layer)
        elif mode == "random":
            target, target_importance, target_layer = find_random_neuron(output, min_input_width, layer=target_layer)
        elif mode == "naive":
            target, target_importance, target_layer = find_naive_neuron(output, layer=target_layer)
        assert target != -1
        # Remove target
        output, new_neuron_index = remove_target_backwards(target, output, new_neuron_index, target_layer)
        # Cull recursively
        curr_neurons = num_hidden_neurons({1: maximal_weights[num_layers - 1]})
    return output, new_neuron_index

def cull_weights_forwards(maximal_weights: dict, neuron_index: dict, model: NIRGraph, hw_config: dict, mode: str):
    max_neurons = hw_config["input"]["num_neurons"]
    curr_neurons = len(maximal_weights[0].keys())
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_neurons > max_neurons:
        if mode == "maximal":
            target_neuron, _ = find_least_important_neuron_layer(output, 0)
        elif mode == "random":
            target_neuron, _, _ = find_random_neuron(output, max_neurons, layer=0)
        elif mode == "naive":
            target_neuron, _, _ = find_naive_neuron(output, layer=0)
        else:
            raise ValueError(f"Unknown")
        output, new_neuron_index = remove_target_forwards(target_neuron, output, new_neuron_index, 0)
        curr_neurons = len(output[0].keys())
    return output, new_neuron_index


def cull_connections(maximal_weights: dict, neuron_index: dict, hw_config: dict, input_bundle: list, output_bundle: list, mode: str):
    max_connections = hw_config["hidden"]["max_connections"]
    min_input_width = hw_config["input"]["num_neurons"]
    # Find num inputs
    num_inputs = len(input_bundle)
    # Find num outputs
    num_outputs = len(output_bundle)
    curr_connections = num_hidden_connections(maximal_weights, num_inputs, num_outputs)
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_connections > max_connections:
        target_neuron, target_layer = -1, -1
        if mode == "maximal":
            target_neuron, target_weight, target_layer = find_least_important_neuron(output)
        elif mode == "random":
            target_neuron, target_weight, target_layer = find_random_neuron(output, min_input_width)
        elif mode == "naive":
            target_neuron, target_weight, target_layer = find_naive_neuron(output)
        assert target_neuron != -1
        output, new_neuron_index = remove_target_backwards(target_neuron, output, new_neuron_index, target_layer)
        curr_connections = num_hidden_connections(output, num_inputs, num_outputs)
    return output, new_neuron_index

def reindex_weights(model: nir.NIRGraph, maximal_weights: dict, neuron_index: dict, input_bundle: list, output_bundle: list):
    out_layers = {}
    num_layers = (len(model.nodes) - 2) // 2
    for i in range(num_layers):
        lif_layer = model.nodes[f"layers_{i * 2 + 1}"]
        if i < num_layers - 1:
            o_indx = np.fromiter(maximal_weights[i+1].keys(), dtype=int)
        else:
            o_indx = np.array(output_bundle)
        i_indx = np.fromiter(maximal_weights[i].keys(), dtype=int)
        if i == 0:
            input_bundle = list(i_indx)
        weights = model.nodes[f"layers_{i * 2}"].weight[o_indx][:, i_indx]
        try:
            bias = model.nodes[f"layers_{i * 2}"].bias[o_indx]
        except AttributeError as e:
            bias = np.zeros((weights.shape[0],))  # Only required if converting to snntorch after
        out_layers[f"layers_{i * 2}"] = (nir.Affine(
            weights,
            bias,
            {'input': np.array([int(len(input_bundle))])},
            {'output': np.array([int(len(output_bundle))])}
        ))
        out_layers[f"layers_{i * 2 + 1}"] = lif_layer
    return out_layers, input_bundle


def split_model_configured(model: nir.NIRGraph, hw_config: dict, split_config: dict, mode: str):
    """
    Core logic for model splitting
    :param mode: one of 'maximal', 'random', 'naive'
    """
    # Handle arguments
    handle_split_arguments(hw_config, split_config)
    # Find I/O split
    num_layers = (len(model.nodes) - 2) // 2
    input_size = model.nodes["input"].input_type["input"]
    if len(input_size) == 5:
        input_size = input_size[:-1]
    # For now, we'll assume the IOs split evenly
    input_bundles = determine_output_bundles(input_size[-1], hw_config)
    num_splits_input = len(input_bundles)
    input_size[-1] /= num_splits_input
    output_size = model.nodes["output"].output_type["output"]
    output_bundles = determine_output_bundles(output_size[-1], hw_config)
    num_splits_output = len(output_bundles)
    output_size[-1] /= num_splits_output
    assert num_splits_input == num_splits_output
    num_models = int(num_splits_output)
    out_weights = [{} for _ in range(num_models)]
    # Initialize output models
    neuron_index = build_neuron_index(model, hw_config, mode=mode)
    min_input_width = hw_config["input"]["num_neurons"]
    for i in range(num_models):
        new_weights_max = find_maximal_valid_weights(neuron_index, output_bundles[i])  # Just for experiment sake
        new_weights_culled, new_neuron_index = cull_weights_backwards(new_weights_max, neuron_index, hw_config["hidden"]["num_neurons"], min_input_width, mode=mode)
        new_weights_culled, new_neuron_index = cull_weights_forwards(new_weights_culled, new_neuron_index, model, hw_config, mode=mode)
        new_weights_culled, new_neuron_index = cull_output_fan_in(new_weights_culled, new_neuron_index, hw_config["output"]["fan_in"], min_input_width, mode=mode)
        new_weights_culled, new_neuron_index = cull_connections(new_weights_culled, new_neuron_index, hw_config, input_bundles[i], output_bundles[i], mode=mode)
        # Rebuild weight matrices
        out_weights[i], input_bundles[i] = reindex_weights(model, new_weights_culled, new_neuron_index, input_bundles[i], output_bundles[i])
        out_weights[i]["input"] = nir.Input({"input": input_size, "bundle": input_bundles[i]})
        out_weights[i]["output"] = nir.Output({"output": output_size, "bundle": output_bundles[i]})
    # Build NIR models
    models = []
    for i, out_weight in enumerate(out_weights):
        models.append(
            nir.NIRGraph(nodes=out_weight, edges=model.edges.copy(), input_type={'input': input_size}, output_type={'output': output_size}, metadata={'input_bundle': input_bundles[i], 'output_bundle': output_bundles[i]})
        )
    return models, input_bundles, output_bundles

def convert_rockpool(model: nir.NIRGraph):
    num_layers = (len(model.nodes) - 2) // 2
    for i in range(num_layers):
        lif_node_name = f"layers_{i * 2 + 1}"
        affine_node_name = f"layers_{i * 2}"
        num_neurons = model.nodes[affine_node_name].weight.shape[0]
        shape = np.array([num_neurons])
        model.nodes[lif_node_name].output_type["output"] = shape
        model.nodes[lif_node_name].input_type["input"] = shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rockpool_model = from_nir(model)
    return rockpool_model


def setup_test_data(data_path: str, batch_size: int, patch_size: int, stride: int, limit: float, encoder_params: dict):
    """
    Sets up data loader, encoder and data module
    TODO: Masked dataloader based on model
    """
    print("Setting up data loader")
    data_source = HeraDeltaNormLoader(
        data_path, limit, patch_size, stride
    )
    # Assuming latency encoding
    encoder = LatencySpikeConverter(
        encoder_params["exposure"],
        encoder_params["tau"],
        encoder_params["normalize"]
    )

    print("Building data module")
    builder = DataModuleBuilder()
    builder.set_dataset(data_source)
    builder.set_encoding(encoder)
    dataset = builder.build(batch_size=batch_size)
    print("Data module ready")
    return dataset, encoder

def test_split(output_dir: str, model_file_path: str, config_file_path: str, patch_size: int, conversion_mode: str):
    # Load example model
    # nir_model = nir.read("/Users/npritchard/PycharmProjects/SNN-RFI-SUPER/lightning_logs/version_171/model.nir")
    # nir_model = nir.read("/Users/npritchard/PycharmProjects/SplitReg/snn-splitreg/FC_LATENCY/LATENCY/HERA/True/32/1.0/lightning_logs/version_0/model.nir")
    # nir_model = nir.read("/Users/npritchard/PycharmProjects/SplitReg/lightning_logs/version_145/model.nir")
    nir_model = nir.read(model_file_path)
    with open(config_file_path, 'r') as ifile:
        orig_config = json.load(ifile)
    # nir_model = nir.read("C:\\Users\\Nicho\\PycharmProjects\\SNN-RFI-SUPER\\lightning_logs\\version_182\\model.nir")
    # Send into split
    models, input_bundles, output_bundles = split_model_configured(nir_model, xylo_hw_config, alg_config, mode=conversion_mode)
    # Load into SNNTorch
    snn_models = []
    for model in models:
        snn_models.append(snntorch.import_from_nir(model))
    # Load into Rockpool
    rockpool_models = []
    xylo_models = []
    for model in models:
        rockpool_models.append(convert_rockpool(model))
        rockpool_model = rockpool_models[-1]
        config, _ = hardware_conversion(rockpool_model)
        xylo_model = setup_xylo(config, dt=1e-4, use_simulator=True)
        xylo_models.append(xylo_model)
    trainer = pl.trainer.Trainer(
        max_epochs=10,
        benchmark=True,
        default_root_dir="./",
        num_nodes=1,
        accelerator="cpu",
        log_every_n_steps=4
    )
    encoder_config = orig_config["encoder"]
    dataset, encoder = setup_test_data("./data", 16, patch_size, patch_size, 1.0, encoder_config)
    hive_model = LitFcMultiplex(snn_models, input_bundles, output_bundles, encoder)
    metrics = trainer.test(hive_model, dataset.test_dataloader())

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
    with open(os.path.join(output_dir, f"{conversion_mode}-metrics.json"), "w") as ofile:
        json.dump(output, ofile, indent=4)

def main():
    base_dir = os.getenv("BASE_DIR")
    model_num = os.getenv("SLURM_ARRAY_TASK_ID")
    patch_size = int(os.getenv("PATCH_SIZE"))
    conversion_mode = os.getenv("CONVERSION_MODE")
    model_dir = os.path.join(base_dir, f"version_{model_num}")
    output_dir = os.path.join(model_dir, "splits")
    os.makedirs(output_dir, exist_ok=True)
    model_file_path = os.path.join(model_dir, "model.nir")
    config_file_path = os.path.join(model_dir, "config.json")
    print(output_dir)
    print(model_file_path)
    test_split(output_dir, model_file_path, config_file_path, patch_size, conversion_mode)


if __name__ == "__main__":
    main()