from rockpool.nn.modules import from_nir, LinearTorch, LIFNeuronTorch
import numpy as np
import nir
import torch

# Supress warnings
import warnings
from tqdm import tqdm
from data.data_loaders import HeraDeltaNormLoader
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import LatencySpikeConverter

# warnings.filterwarnings('ignore')

# Create a NIR graph
nir_model = nir.read("/Users/npritchard/PycharmProjects/SNN-RFI-SUPER/lightning_logs/version_69/model.nir")
# Create Rockpool model from NIR graph.
num_layers = 3
num_input = 8
num_hidden = 128
num_output = 8

with warnings.catch_warnings():
    # Supress unrelated warnings from Rockpool
    warnings.simplefilter("ignore")
    for i in range(num_layers):
        if i == 0:
            input_shape = np.array([num_hidden])
            output_shape = np.array([num_hidden])
        elif i == 1:
            input_shape = np.array([num_hidden])
            output_shape = np.array([num_hidden])
        elif i == 2:
            input_shape = np.array([num_output])
            output_shape = np.array([num_output])
        elif i == 3:
            input_shape = np.array([num_hidden])
            output_shape = np.array([num_hidden])
        elif i == 4:
            input_shape = np.array([num_output])
            output_shape = np.array([num_output])
        # output_shape = nir_model.nodes[f"layers_{i * 2}"].output_type["output"]
        # input_shape = nir_model.nodes[f"layers_{i * 2}"].input_type["input"]
        nir_model.nodes[f"layers_{i * 2 + 1}"].output_type["output"] = output_shape
        nir_model.nodes[f"layers_{i * 2 + 1}"].input_type["input"] = input_shape
    rockpool_model = from_nir(nir_model)
print(rockpool_model)

## Running through some random data (to see what shape is going in/out)
x = torch.randn((36, 4, num_input))
out = rockpool_model(x)
print(out)

for i, layer in enumerate(rockpool_model.modules()):
    print(i)
    if isinstance(layer, LinearTorch):
        # calc weight metrics
        print(f"max: {layer.weight.max().item()}")
        print(f"min: {layer.weight.min().item()}")
        print(f"mean: {layer.weight.mean().item()}")
        print(f"std: {layer.weight.std().item()}")
    elif isinstance(layer, LIFNeuronTorch):
        # calc neuron metrics
        print(f"dt: {layer.dt}")
        print(f"window: {layer.learning_window.item()}")
        print(f"tau_mem: {layer.tau_mem.item()}")
        print(f"threshold: {layer.threshold.item()}")

# Try loading into real hardware
from rockpool.devices.xylo.syns61201 import mapper, config_from_specification, XyloSim
from rockpool.transform.quantize_methods import channel_quantize

spec = mapper(rockpool_model.as_graph())
config, is_valid, msg = config_from_specification(**channel_quantize(**spec))
xylo_model = XyloSim.from_config(config)
example_data = np.random.randint(2, size=(36, 4, num_input))
outputs = []
for example in example_data:
    outputs.append(xylo_model(example)[0])
    xylo_model.reset_state()

# Create data loader
dataset_config = {
    "batch_size": 36
}
encoder_config = {
        "method": "LATENCY",
        "exposure": 16,
        "tau": 1.0,
        "normalize": True,
    }

data_source = HeraDeltaNormLoader(
    "./data", 8, 8, 1.0
)

encoder = LatencySpikeConverter(16, True)

builder = DataModuleBuilder()
builder.set_dataset(data_source)
builder.set_encoding(encoder)
dataset = builder.build(36)


for x, y in tqdm(dataset.test_dset):
    xylo_model.reset_state()
    output = xylo_model(x)
