"""
Main script to run the experiment
"""

import os

from config import get_default_params
from experiment import Experiment


def main():
    model_type = os.getenv("MODEL_TYPE", "FC_LATENCY")
    dataset = os.getenv("DATASET", "HERA")
    plot = os.getenv("PLOT", False) == "True"
    delta_normalization = os.getenv("DELTA_NORMALIZATION", False) == "True"
    patch_size = int(os.getenv("PATCH_SIZE", 64))
    config = get_default_params(
        dataset, model_type, delta_normalization, patch_size
    )
    config["data_source"]["data_path"] = os.getenv(
        "DATA_PATH", config["data_source"]["data_path"]
    )
    config["data_source"]["limit"] = float(
        os.getenv("LIMIT", config["data_source"]["limit"])
    )
    config["encoder"]["exposure"] = int(
        os.getenv("EXPOSURE", config["encoder"].get("exposure", 1))
    )
    config["trainer"]["epochs"] = int(os.getenv("EPOCHS", config["trainer"]["epochs"]))
    print(config)
    root_dir = os.getenv("OUTPUT_DIR", "./")
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    print("Preparation complete")
    experiment.train()
    print("Training complete")
    experiment.evaluate(plot)
    print("Evaluation complete")
    experiment.save_model()


if __name__ == "__main__":
    main()
