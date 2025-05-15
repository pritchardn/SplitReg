"""
This module contains the default configuration parameters for the different models.
"""

import copy
import os

DEFAULT_HERA_LATENCY = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 16,
    },
    "model": {
        "type": "FC_LATENCY",
        "num_inputs": 32,
        "num_hidden": 2048,
        "num_outputs": 32,
        "num_layers": 3,
        "beta": 0.073,
        "alpha": 0.187,
        "fan_in_weighting": 0.06307061193592328,
        "max_connections_weighting": 0.00,
        "max_fan_in": 63,
        "max_connections": 32000,
    },
    "trainer": {
        "epochs": 50,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 14,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_LATENCY_8 = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_8["data_source"]["patch_size"] = 8
DEFAULT_HERA_LATENCY_8["data_source"]["stride"] = 8
DEFAULT_HERA_LATENCY_8["model"]["num_inputs"] = 8
DEFAULT_HERA_LATENCY_8["model"]["num_outputs"] = 8
DEFAULT_HERA_LATENCY_8["model"]["num_hidden"] = 64
DEFAULT_HERA_LATENCY_8["model"]["num_layers"] = 4
DEFAULT_HERA_LATENCY_8["model"]["alpha"] = 0.1273135936989377
DEFAULT_HERA_LATENCY_8["model"]["beta"] = 0.26147990022707174
DEFAULT_HERA_LATENCY_8["model"]["fan_in_weighting"] = 0.042038463798513614
DEFAULT_HERA_LATENCY_8["encoder"]["exposure"] = 62

DEFAULT_HERA_LATENCY_64 = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_64["data_source"]["patch_size"] = 64
DEFAULT_HERA_LATENCY_64["data_source"]["stride"] = 64
DEFAULT_HERA_LATENCY_64["model"]["num_inputs"] = 64
DEFAULT_HERA_LATENCY_64["model"]["num_outputs"] = 64
DEFAULT_HERA_LATENCY_64["model"]["num_hidden"] = 512
DEFAULT_HERA_LATENCY_64["model"]["num_layers"] = 4
DEFAULT_HERA_LATENCY_64["model"]["alpha"] = 0.0008559034957763834
DEFAULT_HERA_LATENCY_64["model"]["beta"] = 0.19808868943799704
DEFAULT_HERA_LATENCY_64["model"]["fan_in_weighting"] = 0.06423761926212247
DEFAULT_HERA_LATENCY_64["encoder"]["exposure"] = 7

DEFAULT_HERA_LATENCY_128 = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_128["data_source"]["patch_size"] = 128
DEFAULT_HERA_LATENCY_128["data_source"]["stride"] = 128
DEFAULT_HERA_LATENCY_128["model"]["num_inputs"] = 128
DEFAULT_HERA_LATENCY_128["model"]["num_outputs"] = 128
DEFAULT_HERA_LATENCY_128["model"]["num_hidden"] = 512
DEFAULT_HERA_LATENCY_128["model"]["num_layers"] = 3
DEFAULT_HERA_LATENCY_128["model"]["alpha"] = 0.14988516044817463
DEFAULT_HERA_LATENCY_128["model"]["beta"] = 0.05971209207088873
DEFAULT_HERA_LATENCY_128["model"]["fan_in_weighting"] = 0.07408406972136698
DEFAULT_HERA_LATENCY_128["encoder"]["exposure"] = 11

DEFAULT_HERA_LATENCY_256 = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_256["data_source"]["patch_size"] = 256
DEFAULT_HERA_LATENCY_256["data_source"]["stride"] = 256
DEFAULT_HERA_LATENCY_256["model"]["num_inputs"] = 256
DEFAULT_HERA_LATENCY_256["model"]["num_outputs"] = 256
DEFAULT_HERA_LATENCY_256["model"]["num_hidden"] = 1024
DEFAULT_HERA_LATENCY_256["model"]["num_layers"] = 3
DEFAULT_HERA_LATENCY_256["model"]["alpha"] = 0.16563137290832453
DEFAULT_HERA_LATENCY_256["model"]["beta"] = 0.4141873600044096
DEFAULT_HERA_LATENCY_256["model"]["fan_in_weighting"] = 0.009322970028350063
DEFAULT_HERA_LATENCY_256["encoder"]["exposure"] = 21

DEFAULT_HERA_LATENCY_512 = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_512["data_source"]["patch_size"] = 512
DEFAULT_HERA_LATENCY_512["data_source"]["stride"] = 512
DEFAULT_HERA_LATENCY_512["model"]["num_inputs"] = 512
DEFAULT_HERA_LATENCY_512["model"]["num_outputs"] = 512
DEFAULT_HERA_LATENCY_512["model"]["num_hidden"] = 512
DEFAULT_HERA_LATENCY_512["model"]["num_layers"] = 4
DEFAULT_HERA_LATENCY_512["model"]["alpha"] = 0.4127688209226251
DEFAULT_HERA_LATENCY_512["model"]["beta"] = 0.027237890368041095
DEFAULT_HERA_LATENCY_512["model"]["fan_in_weighting"] = 0.02049830970257968
DEFAULT_HERA_LATENCY_512["encoder"]["exposure"] = 2

DEFAULT_LOFAR_LATENCY = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_LOFAR_LATENCY["data_source"]["dataset"] = "LOFAR"
DEFAULT_LOFAR_LATENCY["model"]["beta"] = 0.148676888253826
DEFAULT_LOFAR_LATENCY["model"]["num_hidden"] = 512
DEFAULT_LOFAR_LATENCY["model"]["num_layers"] = 2
DEFAULT_LOFAR_LATENCY["encoder"]["exposure"] = 61

DEFAULT_HERA_LATENCY_MH = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 512,
        "stride": 512,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 32,
    },
    "model": {
        "type": "MH_LATENCY",
        "num_inputs": 512,
        "num_hidden": 2048,
        "num_outputs": 512,
        "num_hidden_layers": 2,
        "alpha": 0.99,
        "beta": 0.99,
        "head_width": 16,
        "head_stride": 16,
        "learning_rate": 1e-3,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 10,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_LATENCY_DIVNORM = copy.deepcopy(DEFAULT_HERA_LATENCY)
# DEFAULT_HERA_LATENCY_DIVNORM["model"]["num_hidden"] = 256
# DEFAULT_HERA_LATENCY_DIVNORM["model"]["beta"] = 0.239039586173328
# DEFAULT_HERA_LATENCY_DIVNORM["model"]["num_layers"] = 5
DEFAULT_HERA_LATENCY_DIVNORM["data_source"]["delta_normalization"] = True
# DEFAULT_HERA_LATENCY_DIVNORM["encoder"]["exposure"] = 4

def get_default_params(
        dataset: str,
        model_type: str,
        model_size: int = 128,
        exposure_mode: str = None,
        delta_normalization: bool = False,
        patch_size: int = 32,
):
    if dataset == "HERA":
        if model_type == "FC_LATENCY":
            params = DEFAULT_HERA_LATENCY
            if patch_size == 32:
                params = DEFAULT_HERA_LATENCY
            elif patch_size == 8:
                params = DEFAULT_HERA_LATENCY_8
            elif patch_size == 64:
                params = DEFAULT_HERA_LATENCY_64
            elif patch_size == 128:
                params = DEFAULT_HERA_LATENCY_128
            elif patch_size == 256:
                params = DEFAULT_HERA_LATENCY_256
            elif patch_size == 512:
                params = DEFAULT_HERA_LATENCY_512
            if delta_normalization:
                params["data_source"]["delta_normalization"] = True
        elif model_type == "FC_LATENCY_XYLO":
            if delta_normalization:
                params = DEFAULT_HERA_LATENCY_DIVNORM
            else:
                params = DEFAULT_HERA_LATENCY
            params["model"]["num_inputs"] = 8
            params["model"]["num_outputs"] = 8
            params["model"]["num_layers"] = 3
            params["model"]["num_hidden"] = 128
            params["data_source"]["patch_size"] = 8
            params["data_source"]["stride"] = 8
            raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    params["model"]["type"] = model_type
    params["data_source"]["delta_normalization"] = delta_normalization
    return params
