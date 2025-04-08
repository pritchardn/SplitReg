import nir
import numpy as np
import snntorch
import warnings
from nir import NIRGraph
from rockpool.nn.modules import from_nir

from src.hardware.conversion_example import hardware_conversion, setup_xylo

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

def build_neuron_index(model: nir.NIRGraph, hw_config: dict):
    num_layers = (len(model.nodes) - 2) // 2
    output = {i:{} for i in range(num_layers)}
    fan_out = hw_config["hidden"]["fan_in"]
    for layer in range(num_layers-1, -1, -1):
        weight_layer = model.nodes[f"layers_{layer * 2}"]
        for i, weight in enumerate(weight_layer.weight[:]):
            curr_fan_out = min(fan_out, weight.shape[0])
            # find best k weights
            best_inputs_indx = np.argpartition(np.abs(weight), -curr_fan_out)[-curr_fan_out:]
            best_inputs = weight[best_inputs_indx]
            for neighbour, w in zip(best_inputs_indx, best_inputs):
                output[layer].setdefault(i, dict())[int(neighbour)] = w
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

def cull_weights_backwards(maximal_weights: dict, neuron_index: dict, model: NIRGraph, hw_config: dict):
    max_neurons = hw_config["hidden"]["num_neurons"]
    num_layers = len(neuron_index.keys())
    curr_neurons = num_hidden_neurons(maximal_weights)
    if curr_neurons < max_neurons:
        return maximal_weights, neuron_index
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_neurons > max_neurons:
        # Find least important neuron
        target, target_importance, target_layer = find_least_important_neuron(output)
        assert target != -1
        # Remove target
        output, new_neuron_index = remove_target_backwards(target, output, new_neuron_index, target_layer)
        # Cull recursively
        curr_neurons = num_hidden_neurons(output)
    return output, new_neuron_index

def cull_weights_forwards(maximal_weights: dict, neuron_index: dict, model: NIRGraph, hw_config: dict):
    max_neurons = hw_config["input"]["num_neurons"]
    curr_neurons = len(maximal_weights[0].keys())
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_neurons > max_neurons:
        target_neuron, target_weight = find_least_important_neuron_layer(output, 0)
        output, new_neuron_index = remove_target_forwards(target_neuron, output, new_neuron_index, 0)
        curr_neurons = len(output[0].keys())
    return output, new_neuron_index


def cull_connections(maximal_weights: dict, neuron_index: dict, hw_config: dict, input_bundle: list, output_bundle: list):
    max_connections = hw_config["hidden"]["max_connections"]
    # Find num inputs
    num_inputs = len(input_bundle)
    # Find num outputs
    num_outputs = len(output_bundle)
    curr_connections = num_hidden_connections(maximal_weights, num_inputs, num_outputs)
    output = maximal_weights.copy()
    new_neuron_index = neuron_index.copy()
    while curr_connections > max_connections:
        target_neuron, target_weight, target_layer = find_least_important_neuron(output)
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
    return out_layers


def split_model_configured(model: nir.NIRGraph, hw_config: dict, split_config: dict):
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
    neuron_index = build_neuron_index(model, hw_config)
    for i in range(num_models):
        new_weights_max = find_maximal_valid_weights(neuron_index, output_bundles[i])  # Just for experiment sake
        new_weights_culled, new_neuron_index = cull_weights_backwards(new_weights_max, neuron_index, model, hw_config)
        new_weights_culled, new_neuron_index = cull_weights_forwards(new_weights_culled, new_neuron_index, model, hw_config)
        new_weight_culled, new_neuron_index = cull_connections(new_weights_culled, new_neuron_index, hw_config, input_bundles[i], output_bundles[i])
        # Rebuild weight matrices
        out_weights[i] = reindex_weights(model, new_weights_culled, new_neuron_index, input_bundles[i], output_bundles[i])
        out_weights[i]["input"] = nir.Input({"input": input_size, "bundle": input_bundles[i]})
        out_weights[i]["output"] = nir.Output({"output": output_size, "bundle": output_bundles[i]})
    # Build NIR models
    models = []
    for out_weight in out_weights:
        models.append(
            nir.NIRGraph(nodes=out_weight, edges=model.edges.copy(), input_type={'input': input_size}, output_type={'output': output_size}, metadata={})
        )
    return models

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

if __name__ == "__main__":
    # Load example model
    nir_model = nir.read("/Users/npritchard/PycharmProjects/SNN-RFI-SUPER/lightning_logs/version_171/model.nir")
    # nir_model = nir.read("C:\\Users\\Nicho\\PycharmProjects\\SNN-RFI-SUPER\\lightning_logs\\version_182\\model.nir")
    # Send into split
    models  = split_model_configured(nir_model, xylo_hw_config, alg_config)
    # Load into SNNTorch
    snn_models = []
    for model in models:
        snn_models.append(snntorch.import_from_nir(model))
    # Load into Rockpool
    rockpool_models = []
    for model in models:
        rockpool_models.append(convert_rockpool(model))
        rockpool_model = rockpool_models[-1]
        config, _ = hardware_conversion(rockpool_model)
        xylo_model = setup_xylo(config, dt=1e-4, use_simulator=True)
