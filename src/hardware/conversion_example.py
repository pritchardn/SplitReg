import numpy as np
import nir
import samna
import torch
import warnings
from tqdm import tqdm
from time import sleep

# Rockpool imports
from rockpool.nn.modules import from_nir, LinearTorch, LIFNeuronTorch
from rockpool.devices.xylo.syns61201 import mapper, config_from_specification, XyloSim
from rockpool.transform.quantize_methods import channel_quantize, global_quantize
from rockpool.devices.xylo.helper import find_xylo_hdks

# Project-specific imports
from data.data_loaders import HeraDeltaNormLoader
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import LatencySpikeConverter
from evaluation import calculate_metrics

# --- Configuration Constants ---
# TODO: These should ideally be arguments or read from a config file
NIR_MODEL_PATH = "/Users/npritchard/PycharmProjects/SNN-RFI-SUPER/lightning_logs/version_170/model.nir"
DATA_PATH = "./data"
USE_SIMULATOR = True

# Model architecture parameters (should match the loaded NIR model)
NUM_LAYERS = 3
NUM_INPUT = 8
NUM_HIDDEN = 63  # Based on Xylo Hardware Specs
NUM_OUTPUT = 8

# Data parameters
BATCH_SIZE = 36 # Batch size for evaluation (Note: original code used 1, but random data test used 36)
TEST_BATCH_SIZE = 36 # Batch size during hardware/sim inference loop
DATA_SPLIT_RATIO = 0.1  # TODO: Change names here
PATCH_SIZE = 8
STRIDE = 8

# Encoder parameters
ENCODER_METHOD = "LATENCY"
ENCODER_EXPOSURE = 16
ENCODER_TAU = 1.0
ENCODER_NORMALIZE = True

# Hardware / Simulation parameters
SIMULATION_DT = 1e-4

def load_and_convert_model(nir_path: str, num_layers: int, num_hidden: int, num_output: int):
    """
    Loads a NIR model, adjusting node shapes to fit Rockpool model.
    The actual adjustment code is currently very fragile.
    :param nir_path: Path to the NIR model (saved by SnnTorch)
    :param num_layers: Number of Linear-LIF pairs
    :param num_hidden: Number of neurons in hidden layers
    :param num_output: Number of neurons in the output layer
    :return: Converted rockpool model
    """
    print(f"Loading NIR model from {nir_path}")
    nir_model = nir.read(nir_path)
    print("Fixing NIR node shapes")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            for i in range(num_layers):
                lif_node_name = f"layers_{i * 2 + 1}"
                if i == 0:
                    shape = np.array([num_hidden])
                elif i == num_layers - 1:
                    shape = np.array([num_output])
                else:
                    shape = np.array([num_hidden])
                if lif_node_name in nir_model.nodes:
                    print(f"Adjusting {lif_node_name} to {shape}")
                    nir_model.nodes[lif_node_name].output_type["output"] = shape
                    nir_model.nodes[lif_node_name].input_type["input"] = shape
                else:
                    print(f"Expected {lif_node_name} not found in NIR model")
        except KeyError as e:
            print(f"Could not access node during shape adjustment {e}")
            raise
        except AttributeError as e:
            print(f"Error setting node shape {e}")
            raise
    print("Converting NIR graph to Rockpool model")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        rockpool_model = from_nir(nir_model)
    print("Conversion successful")
    print(rockpool_model)
    return rockpool_model

def analyse_rockpool_model(rockpool_model):
    """
    Iterates through the model and prints metrics for Linear and LIF layers
    :param rockpool_model:
    """
    print("Analysing rockpool model")
    for i, layer in enumerate(rockpool_model.modules()):
        print(f"Layer {i}: {layer}")
        if isinstance(layer, LinearTorch):
            print("Type: Linear Layer")
            print(f"  Weight Stats:")
            print(f"    Max:  {layer.weight.max().item():.4f}")
            print(f"    Min:  {layer.weight.min().item():.4f}")
            print(f"    Mean: {layer.weight.mean().item():.4f}")
            print(f"    Std:  {layer.weight.std().item():.4f}")
            if layer.bias is not None:
                print(f"  Bias Stats:")
                print(f"    Max:  {layer.bias.max().item():.4f}")
                print(f"    Min:  {layer.bias.min().item():.4f}")
                print(f"    Mean: {layer.bias.mean().item():.4f}")
                print(f"    Std:  {layer.bias.std().item():.4f}")
            else:
                print("  Bias: None")
        elif isinstance(layer, LIFNeuronTorch):
            print("  Type: LIF Neuron Layer")
            # Ensure parameters are tensors and extract scalar values safely
            tau_mem = layer.tau_mem.item() if isinstance(layer.tau_mem, torch.Tensor) else layer.tau_mem
            tau_syn = layer.tau_syn.item() if isinstance(layer.tau_syn, torch.Tensor) else layer.tau_syn
            threshold = layer.threshold.item() if isinstance(layer.threshold, torch.Tensor) else layer.threshold
            print(f"  Neuron Params:")
            print(f"    dt:        {layer.dt}")
            print(f"    tau_mem:   {tau_mem:.4f}")
            print(f"    tau_syn:   {tau_syn:.4f}")
            print(f"    threshold: {threshold:.4f}")

def hardware_conversion(rockpool_model):
    """
    Maps the Rockpool model graph to hardware specifications and prepares it for deployment
    in a simulator or hardware.
    :param rockpool_model:
    :return: (config, spec) - Config contains hardware configuration, spec contains specification before quantization
    """
    print("Preparing for hardware/simulation")
    graph = rockpool_model.as_graph()
    spec = mapper(graph)
    print("Applying channel quantization")
    # spec["threshold_out"] *= 1
    quantized_spec = channel_quantize(**spec)
    print("Generating hardware configuration")
    config, is_valid, msg = config_from_specification(**quantized_spec)
    assert is_valid, f"Generated hardware config is not valid: {msg}"
    print("Hardware generation successful")
    return config, spec

def setup_xylo(config, dt, use_simulator=True):
    """
    Sets up either Xylo hardware interface or simulator
    :param config: Hardware configuration dictionary
    :param dt: Simulation timestep
    :param use_simulator: If true, use simulator, try to use hardware if false
    :return: XyloSim or XyloSamna if simulator or hardware respectively
    """
    if use_simulator:
        print("Setting up simulator")
        xylo_model = XyloSim.from_config(config, dt=dt)
        print("Simulator ready")
        return xylo_model
    else:
        print("Setting up hardware")
        hdks, mods, _ = find_xylo_hdks()
        if not hdks:
            raise ValueError("Could not find any connected HDKs")
        hdk = hdks[0]
        mod = mods[0]
        xylo_model = mod.XyloSamna(hdk, config=config, dt=dt)
        print("Xylo Hardware interface ready")
        return xylo_model

def setup_test_data(data_path: str, batch_size: int, patch_size: int, stride: int, limit: float, encoder_params: dict):
    """
    Sets up data loader, encoder and data module
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


def evaluate_model(xylo_model, dataset, encoder):
    """
    Runs the evaluation loop on the test dataset using the hardware/simulator module.
    It is testing for accuracy
    """
    print("Starting evaluation")
    accuracy, mse, auroc, auprc, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    test_loader = dataset.val_dataloader()
    dataset_len = len(test_loader)
    if dataset_len == 0:
        return {"accuracy": accuracy, "mse": mse, "auroc": auroc, "auprc": auprc, "f1": f1}

    accuracy, mse, auroc, auprc, f1 = 0.0, 0.0, 0.0, 0.0, 0.0
    for x, y in tqdm(test_loader, desc="Evaluating"):
        big_output = []
        x = x.detach().cpu().numpy().astype(np.int16)
        for ex in x:
            output = []
            for t in range(x.shape[-1]):
                output.append(xylo_model(ex[..., t].squeeze(1))[0])
            big_output.append(np.stack(output, axis=-1))
        xylo_model.reset_state()
        big_output = np.stack(big_output, axis=0)
        # Calculate stats on batch
        # Decode inference
        big_output = np.expand_dims(np.moveaxis(big_output, 1, 0), axis=2)
        output_pred = encoder.decode_inference(big_output)
        b_accuracy, b_mse, b_auroc, b_auprc, b_f1 = calculate_metrics(
            y.detach().cpu().numpy(), output_pred
        )
        accuracy += b_accuracy
        mse += b_mse
        auroc += b_auroc
        auprc += b_auprc
        f1 += b_f1
    print(dataset_len)
    print(accuracy, mse, auroc, auprc, f1)
    accuracy /= dataset_len
    mse /= dataset_len
    auroc /= dataset_len
    auprc /= dataset_len
    f1 /= dataset_len
    return {"accuracy": accuracy, "mse": mse, "auroc": auroc, "auprc": auprc, "f1": f1}

def evaluate_idle_power(xylo_model):
    """
    Evaluates the idle power consumption of real xylo_hardware
    """
    print("Evaluating idle power consumption")
    xylo_model._power_buf.get_events()
    sleep(5.)
    power = xylo_model._power_buf.get_events()
    power_idle = ([], [], [], [])
    for p in power:
        power_idle[p.channel].append(p.value)
    idle_power_per_channel = np.mean(np.stack(power_idle), axis=1)
    channels = samna.xyloA2TestBoard.MeasurementChannels
    io_power = idle_power_per_channel[channels.Io]
    afe_core_power = idle_power_per_channel[channels.LogicAfe]
    afe_ldo_power = idle_power_per_channel[channels.IoAfe]
    snn_core_power = idle_power_per_channel[channels.Logic]
    print(f'XyloAudio 2\nAll IO:\t\t{io_power * 1e6:.1f} µW\nAFE core:\t{afe_core_power * 1e6:.1f} µW\nInternal LDO:\t{afe_ldo_power * 1e6:.1f} µW\nSNN core logic:\t{snn_core_power*1e6:.1f} µW')
    return idle_power_per_channel

def evaluate_power(xylo_model, dataset, encoder):
    print("Evaluating active power consumption")
    xylo_model._power_buf.get_events()
    # Run through inference TODO: make batched
    for x, y in tqdm(dataset.test_dataloader(), desc="Evaluating"):
        x = x.detach().cpu().numpy().astype(np.int16)
        for ex in x:
            output = []
            for t in range(x.shape[-1]):
                output.append(xylo_model(ex[..., t].squeeze(1))[0])
    power = xylo_model._power_buf.get_events()
    power_active = ([], [], [], [])
    for p in power:
        power_active[p.channel].append(p.value)
    active_power_per_channel = np.mean(np.stack(power_active), axis=1)
    channels = samna.xyloA2TestBoard.MeasurementChannels
    io_power = active_power_per_channel[channels.Io]
    afe_core_power = active_power_per_channel[channels.LogicAfe]
    afe_ldo_power = active_power_per_channel[channels.IoAfe]
    snn_core_power = active_power_per_channel[channels.Logic]
    print(f'XyloAudio 2\nAll IO:\t\t{io_power * 1e6:.1f} µW\nAFE core:\t{afe_core_power * 1e6:.1f} µW\nInternal LDO:\t{afe_ldo_power * 1e6:.1f} µW\nSNN core logic:\t{snn_core_power*1e6:.1f} µW')
    return active_power_per_channel

def main():
    rockpool_model = load_and_convert_model(NIR_MODEL_PATH, NUM_LAYERS, NUM_HIDDEN, NUM_OUTPUT)
    analyse_rockpool_model(rockpool_model)
    config, _ = hardware_conversion(rockpool_model)
    xylo_model = setup_xylo(config, dt=SIMULATION_DT, use_simulator=True)
    encoder_config = {
        "method": "LATENCY",
        "exposure": ENCODER_EXPOSURE,
        "tau": ENCODER_TAU,
        "normalize": ENCODER_NORMALIZE,
    }
    dataset, encoder = setup_test_data(DATA_PATH, TEST_BATCH_SIZE, PATCH_SIZE, STRIDE, DATA_SPLIT_RATIO, encoder_config)
    metrics = evaluate_model(xylo_model, dataset, encoder)
    print("\n--- Final Evaluation Metrics ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"MSE:      {metrics['mse']:.4f}")
    print(f"AUROC:    {metrics['auroc']:.4f}")
    print(f"AUPRC:    {metrics['auprc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print("------------------------------")



if __name__ == "__main__":
    main()