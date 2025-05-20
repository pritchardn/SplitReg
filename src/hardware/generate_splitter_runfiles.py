import os

scratch_path = "/scratch/pawsey0411/npritchard"
software_path = "/software/projects/pawsey0411/npritchard/setonix/2023.08/python"
hpc_account = "pawsey0411"

PATCH_SIZES = [8, 32, 64, 128, 256, 512]
CONVERSION_METHODS = ["maximal", "naive", "random"]


def write_bashfile(out_dir, name, runfiletext):
    with open(os.path.join(out_dir, f"{name}.sh"), "w") as f:
        f.write(runfiletext)


def prepare_conversion(target_dir: str, patch_size: int, conversion_method: str, num_models: int):
    runfiletext = (
        f"""#!/bin/bash
#SBATCH --job-name=SNN-SPLITREG-{patch_size}-{conversion_method}
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-{num_models - 1}
#SBATCH --partition=gpu
#SBATCH --account={hpc_account}-gpu
#SBATCH --gres=gpu:1

export DATA_PATH="{scratch_path}/data"
export BASE_DIR="{target_dir}"
export PATCH_SIZE={patch_size}
export CONVERSION_MODE={conversion_method}

module load python/3.10.10

cd {sofware_path}/SNN-SPLITREG/src
source {software_path}/snn-nln/bin/activate

export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_PLACES=cores     
export OMP_PROC_BIND=close  

srun -N 1 -n 1 -c 8 --gpus-per-task=1 --gpu-bind=closest python3 splitter_main.py
"""
    )
    return runfiletext


def main(out_dir: str, in_dir: str, num_models=10):
    os.makedirs(out_dir, exist_ok=True)
    for patch_size in PATCH_SIZES:
        target_dir = os.path.join(in_dir, str(patch_size), "1.0", "lightning_logs")
        for conversion_method in CONVERSION_METHODS:
            out_filename = f"FC_LATENCY_SPLIT-{patch_size}-{conversion_method}"
            write_bashfile(
                out_dir,
                out_filename,
                prepare_conversion(target_dir, patch_size, conversion_method, num_models)
            )


if __name__ == "__main__":
    in_dir = f"{scratch_path}/outputs/snn-splitreg/FC_LATENCY/LATENCY/HERA/True/"
    main(f"./src/hpc/SPLITS", in_dir)
