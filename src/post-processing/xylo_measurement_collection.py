import os
import json
from trial_results_processing import process_metric_file, write_output_csv, write_output_json, write_summary_csv, \
    calculate_results_summary


def main_power_splitters(in_dirname: str, out_dirname: str, out_filename: str = "summary", limit: int = None, power = 50):
    # Collect all metrics.json files in directory and subdirectories
    max_metrics_files, naive_metrics_files, random_metrics_files = [], [], []
    max_results, naive_results, random_results = [], [], []
    max_count, naive_count, random_count = 0, 0, 0
    print(in_dirname)
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if file == f"maximal-{power}-power_metrics.json":
                max_metrics_files.append(os.path.join(root, file))
                max_results.append(process_metric_file(os.path.join(root, file))["active"])
                max_count += 1
            elif file == f"naive-{power}-power_metrics.json":
                naive_metrics_files.append(os.path.join(root, file))
                naive_results.append(process_metric_file(os.path.join(root, file))["active"])
                naive_count += 1
            elif file == f"random-{power}-power_metrics.json":
                random_metrics_files.append(os.path.join(root, file))
                random_results.append(process_metric_file(os.path.join(root, file))["active"])
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
        write_output_csv(os.path.join(out_dirname, f"{name}-{out_filename}-{power}.csv"), results)
        write_output_json(os.path.join(out_dirname, f"{name}-{out_filename}-{power}.json"), summary)
        write_summary_csv(os.path.join(out_dirname, f"{name}-{out_filename}-{power}_summary.csv"), summary)


def main_process_supercomputer():
    for model, encoding, size in [
        ("FC_LATENCY_REG", "LATENCY", "8"),
        ("FC_LATENCY_REG", "LATENCY", "32"),
        ("FC_LATENCY_REG", "LATENCY", "64"),
        ("FC_LATENCY_REG", "LATENCY", "128"),
        ("FC_LATENCY_REG", "LATENCY", "256"),
        ("FC_LATENCY_REG", "LATENCY", "512"),
    ]:
        root_dir = f".{os.sep}snn-splitreg{os.sep}{model}{os.sep}{encoding}{os.sep}HERA{os.sep}True{os.sep}{size}{os.sep}1.0"
        log_dir = f"lightning_logs{os.sep}"
        output_dir = f".{os.sep}"
        for power in [50, 6.25]:
            main_power_splitters(os.path.join(root_dir, log_dir), output_dir, f"{model}-{size}-{power}", limit=10, power=power)

if __name__ == "__main__":
    main_process_supercomputer()