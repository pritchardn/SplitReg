# Maximal Split
This work demonstrates the training and splitting of SNNs for RFI detection in radio astronomy for inference on SynSense
Xylo hardware. 
Training in snnTorch, model manipulation in NIR before inference in Rockpool/Xylo demonstrates the usefuless of 
recently developed interoperability tools.
Energy estimates and on-hardware inference (if available) show power consumption per-chip of ~250 micro-watts.

This codebase provides the ability to search for hyperparameters, train models, perform splitting and estimate energy usage. 

## Introduction

### Key Components

- Supervised training of SNNs for RFI detection in radio astronomy with latency encoding
- Maximal splitting algorithm for model sharding in NIR format
- Combined inference in snnTorch or deployment to SynSense Xylo hardware (if available)

### System Overview

<img src="__assets__/example.png" width="1000px">

Overall methodology for RFI detection with spiking neural networks trained as a large single model and then split for inference on several neuromorphic chipsets. Spectrograms are split and latency encoded before feeding through the SNN, models are split in NIR format and deployed in snnTorch or to SynSense Xylo hardware for power measurement.

## Code Structure
We outline several important files in the code-structure below
```
data/
├── HERA-21-11-2024_all_delta_norm.pkl # [Available online](https://zenodo.org/records/14676274)
src/
├── data/
│   ├── data_loaders.py  # Contains files that read in raw data
├── hardware/
    ├── conversion_example.py  # Contains Xylo deployment code from NIR, includes a small stand-alone example
├── hpc/
    ├── generate_runfiles.py  # Generates slurm-scripts for HPC environment
├── interfaces/
    ├── # Various boilerplate interfaces used elsewhere
├── models/
    ├── fc_latency.py  # Original sized SNN model for training
    ├── fc_multiplex.py  # Lightning model comprised of several split snnTorch models
    ├── fc_hivemind.py  # Lightning model comprised of several split Rockpool models
├── post-processing/
    ├── energy_estimates.py  # Code to generate energy estimates from analytical model
    ├── trial_results_processing.py  # Code to generate performance summaries from repeat snnTorch trials
    ├── xylo_measurement_collection.py  # Code to generate performance summaries from repeat Xylo trials
├── config.py  # Contains training hyper-parameters
├── evlauation.py  # Contains boilerplate to run through and plot results post-training
├── generate_plots.py  # Generates rasters of spike encodings
├── main.py  # Main training file
├── optuna_main.py  # Main training file for optuna hyper-parameter trials
```
## Setup
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```
## Training
For the purposes of this readme, we provide scripts to reproduce a complete trial from training, splitting and energy estimates.
`hpc/generate_runfiles.py` can generate a complete set of slurm scripts to run all repeat trials; although this will be 
exhaustive.

To train:

`python`

To perform split inference

`python`

To generate energy estimates

`python`



## License

This project is licensed under the MIT License - see the LICENSE file for details.
