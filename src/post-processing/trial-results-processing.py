"""
This script processes the results of the trials and calculates the summary
statistics for the metrics.
"""

import csv
import json
import os

import numpy as np


def process_metric_file(filename: str):
    with open(filename, "r") as file:
        data = json.load(file)
    data = json.loads(data)
    return data


def calculate_results_summary(results: list):
    metrics = list(set(key for result in results for key in result.keys()))
    summary = {}
    for metric in metrics:
        values = [result[metric] for result in results]  # Want it to error if not found
        summary[metric] = {
            "mean": np.nanmean(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
        }
    summary["n"] = len(results)
    return summary


def write_output_csv(filename: str, data: list):
    with open(filename, "w", newline="") as ofile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def write_output_json(filename: str, data: dict):
    with open(filename, "w") as ofile:
        json.dump(data, ofile, indent=4)


def write_summary_csv(filename: str, data: dict):
    # Convert dict into list
    data_list = []
    num_trials = 0
    for key, value in data.items():
        if isinstance(value, int):
            num_trials = value
            continue
        value["title"] = key
        data_list.append(value)
    for row in data_list:
        row["n"] = num_trials
    print(data_list)
    # Write output
    with open(filename, "w", newline="") as ofile:
        fieldnames = data_list[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)


def main(
    in_dirname: str, out_dirname: str, out_filename: str = "summary", limit: int = None
):
    # Collect all metrics.json files in directory and subdirectories
    metrics_files = []
    results = []
    count = 0
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if file == "metrics.json":
                metrics_files.append(os.path.join(root, file))
                results.append(process_metric_file(os.path.join(root, file)))
                count += 1
        if limit is not None and count >= limit:
            break
    print(len(metrics_files))
    print(json.dumps(results, indent=4))
    summary = calculate_results_summary(results)
    print(json.dumps(summary, indent=4))
    write_output_csv(os.path.join(out_dirname, f"{out_filename}.csv"), results)
    write_output_json(os.path.join(out_dirname, f"{out_filename}.json"), summary)
    write_summary_csv(os.path.join(out_dirname, f"{out_filename}_summary.csv"), summary)

def main_splitters(in_dirname: str, out_dirname: str, out_filename: str = "summary", limit: int = None):
    # Collect all metrics.json files in directory and subdirectories
    max_metrics_files, naive_metrics_files, random_metrics_files = [], [], []
    max_results, naive_results, random_results = [], [], []
    max_count, naive_count, random_count = 0, 0, 0
    print(in_dirname)
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if file == "maximal-metrics.json":
                max_metrics_files.append(os.path.join(root, file))
                max_results.append(process_metric_file(os.path.join(root, file)))
                max_count += 1
            elif file == "naive-metrics.json":
                naive_metrics_files.append(os.path.join(root, file))
                naive_results.append(process_metric_file(os.path.join(root, file)))
                naive_count += 1
            elif file == "random-metrics.json":
                random_metrics_files.append(os.path.join(root, file))
                random_results.append(process_metric_file(os.path.join(root, file)))
                random_count += 1
        if limit is not None and (max_count >= limit or naive_count >= limit or random_count >= limit):
            break
    for metrics_files, results, name in [(max_metrics_files, max_results, "maximal"),
                                        (naive_metrics_files, naive_results, "naive"),
                                        (random_metrics_files, random_results, "random")]:
        print(len(metrics_files))
        print(json.dumps(results, indent=4))
        summary = calculate_results_summary(results)
        print(json.dumps(summary, indent=4))
        write_output_csv(os.path.join(out_dirname, f"{name}-{out_filename}.csv"), results)
        write_output_json(os.path.join(out_dirname, f"{name}-{out_filename}.json"), summary)
        write_summary_csv(os.path.join(out_dirname, f"{name}-{out_filename}_summary.csv"), summary)



def main_process_supercomputer():
    for model, encoding, size in [
        ("FC_LATENCY", "LATENCY", "32"),
        ("FC_LATENCY", "LATENCY", "64"),
        ("FC_LATENCY", "LATENCY", "128"),
        ("FC_LATENCY", "LATENCY", "256"),
        ("FC_LATENCY", "LATENCY", "512"),
    ]:
        root_dir = f".{os.sep}snn-splitreg{os.sep}{model}{os.sep}{encoding}{os.sep}HERA{os.sep}True{os.sep}{size}{os.sep}1.0"
        log_dir = f"lightning_logs{os.sep}"
        output_dir = f".{os.sep}"
        main(os.path.join(root_dir, log_dir), output_dir, f"{model}-{size}", limit=10)


def main_process_custom():
    root_dir = (
        "./snn-super/FC_LATENCY_XYLO/LATENCY/HERA_POLAR_FULL/True/1.0/"
    )
    log_dir = "lightning_logs/"
    output_dir = "./"
    main(os.path.join(root_dir, log_dir), output_dir, "LATENCY_XYLO_HERA_DIVNORM_POLAR_FULL")


if __name__ == "__main__":
    main_process_supercomputer()
