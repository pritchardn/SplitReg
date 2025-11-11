# SplitReg
A demonstration of my regularisation and SNN splitting techniques.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17576919.svg)](https://doi.org/10.5281/zenodo.17576919)

Contains:
- Implementation of supervised SNN training via snnTorchfor RFI detection
- Implementation of model splitting methods and driver code for SynSense Rockpool and Xylo deployment

All code files are in `src/`.

## Installation
```bash
conda create -n splitreg python=3.10
conda activate splitreg
pip install -r src/requirements.txt
```
You may need extra instructions for installing PyTorch with your specific CPU/GPU combination.

### Data Dependencies
The data used in this project is not included in this repository.
You will need to download the HERA dataset from [zenodo](https://zenodo.org/record/6724065) and unzip
them into `/data`.

## Licensing
This code is licensed under the MIT License. See the LICENSE file for details.
