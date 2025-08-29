# Save this file as run_baselines.py
import time
import json
import os
import subprocess
import atexit
import shlex
import requests # Import requests for the health check
import shutil   # Import for directory operations
from datetime import datetime # Import the datetime class

def wait_for_server_ready(url, timeout=10000):
    """
    Polls the server to check if it's ready, until a timeout is reached.
    """
    print(f"Waiting for server to be ready at {url}...", end="")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            requests.get(url, timeout=5)
            print("\nServer is ready!")
            return True
        except requests.exceptions.ConnectionError:
            print(".", end="", flush=True)
            time.sleep(2)
        except requests.exceptions.ReadTimeout:
            print("\nServer is responsive (timed out), considering it ready.")
            return True
    
    print(f"\nError: Server was not ready within {timeout} seconds.")
    return False

def analyze_baseline_metrics(metrics_file="baseline_metrics_8003.jsonl"):
    """Reads the baseline metrics log and computes total time and TPS for each model."""
    timings = {"qwen": 0.0, "dream": 0.0}
    counts = {"qwen": 0, "dream": 0}
    total_tokens = {"qwen": 0, "dream": 0}

    if not os.path.exists(metrics_file):
        return {}

    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                metric = json.loads(line)
                model = metric.get("model")
                if model in timings:
                    timings[model] += metric.get("time_s", 0)
                    counts[model] += 1
                    total_tokens[model] += metric.get("generated_tokens", 0)
            except json.JSONDecodeError:
                continue
    
    qwen_tps = (total_tokens["qwen"] / timings["qwen"]) if timings["qwen"] > 0 else 0
    dream_tps = (total_tokens["dream"] / timings["dream"]) if timings["dream"] > 0 else 0

    return {
        "qwen_total_time_s": timings["qwen"],
        "qwen_requests": counts["qwen"],
        "qwen_tokens_per_second": qwen_tps,
        "dream_total_time_s": timings["dream"],
        "dream_requests": counts["dream"],
        "dream_tokens_per_second": dream_tps,
    }

def run_single_baseline(config, model_type, benchmark):
    """
    Runs lm-harness for a single baseline model on a single benchmark.
    [MODIFIED] This function now also handles saving results to the final log directory.
    """
    
    endpoint_map = {
        "qwen": "v1/qwen/chat/completions",
        "dream": "v1/dream/chat/completions"
    }
    
    model_name_map = {
        "qwen": config["qwen_model_name"],
        "dream": config["dream_model_name"]
    }

    print(f"\n--- Running Evaluation for: {model_type.upper()} on Benchmark: {benchmark} ---")
    
    # Define a temporary directory for lm-harness output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_results_dir = f"temp_results_{model_type}_{benchmark}_{timestamp}"

    model_args_str = (
        f"base_url=http://127.0.0.1:8003/{endpoint_map[model_type]},"
        f"model={model_name_map[model_type]},"
        f"eos_string='<|im_end|>',"
        f"truncate=True"
    )

    command = [
        "python", "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args_str,
        "--tasks", benchmark, # Use the benchmark passed to the function
        "--apply_chat_template",
        "--output_path", temp_results_dir,
        "--batch_size", "1",
        "--num_fewshot", "5",
    ]
    if "limit" in config:
        command.extend(["--limit", str(config["limit"])])

    print(f"Executing command: {' '.join(shlex.quote(c) for c in command)}")
    subprocess.run(command, check=True)

    # [NEW] Create final log directory and save artifacts
    final_log_path = os.path.join("eval_log_baseline", model_type, "128step", benchmark)
    os.makedirs(final_log_path, exist_ok=True)
    print(f"Organizing results into: {final_log_path}")

    # Copy lm-harness results
    if os.path.exists(temp_results_dir):
        # shutil.copytree requires the destination directory not to exist, so we create a sub-folder
        destination_dir = os.path.join(final_log_path, os.path.basename(temp_results_dir))
        shutil.copytree(temp_results_dir, destination_dir)
        shutil.rmtree(temp_results_dir) # Clean up temporary directory
        print(f"Saved lm-harness results to {destination_dir}")

def main():
    """
    Main function to run the entire baseline evaluation pipeline.
    [MODIFIED] Iterates through models and a list of benchmarks.
    """
    EVAL_CONFIG = {
        "qwen_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "dream_model_name": "Dream-org/Dream-v0-Instruct-7B",
        "benchmarks": ["mmlu_pro"], # [MODIFIED] Now a list of benchmarks
        # "limit": 5, # Add limit here for quick testing runs
    }

    metrics_file = "baseline_metrics_8003.jsonl"

    # Start the baseline API server
    server_process = subprocess.Popen(
        ["uvicorn", "baseline_api_server_8003:app", "--host", "0.0.0.0", "--port", "8003"],
    )
    atexit.register(server_process.terminate)
    print(f"Baseline API server started with PID: {server_process.pid}. Waiting...")
    
    health_check_url = "http://127.0.0.1:8003/v1/qwen/chat/completions"
    if not wait_for_server_ready(health_check_url):
        print("Script terminating because the server failed to start.")
        server_process.terminate()
        server_process.wait()
        exit(1)

    # [MODIFIED] Loop through models and benchmarks
    models_to_evaluate = ['dream']
    for model_type in models_to_evaluate:
        for benchmark in EVAL_CONFIG["benchmarks"]:
            # Clear metrics file before each run
            if os.path.exists(metrics_file):
                os.remove(metrics_file)

            run_single_baseline(EVAL_CONFIG, model_type, benchmark)

            # [NEW] Analyze metrics and save the report immediately after the run
            final_metrics = analyze_baseline_metrics(metrics_file)
            
            # final_log_path = os.path.join("eval_log_baseline", model_type, benchmark)
            final_log_path = os.path.join("eval_log_baseline", model_type, "128step", benchmark)

            report_path = os.path.join(final_log_path, "metrics_report.txt")
            
            with open(report_path, 'w') as f:
                print(f"--- Performance Report for {model_type.upper()} on {benchmark} ---", file=f)
                if final_metrics:
                    if final_metrics.get(f'{model_type}_requests', 0) > 0:
                        report_line = (
                            f"{model_type.capitalize()} Baseline ({final_metrics[f'{model_type}_requests']} requests): "
                            f"{final_metrics[f'{model_type}_total_time_s']:.2f} s | "
                            f"Tokens/s: {final_metrics[f'{model_type}_tokens_per_second']:.2f}"
                        )
                        print(report_line) # Print to console
                        print(report_line, file=f) # Print to file
                else:
                    print("No baseline metrics were logged.", file=f)
            print(f"Saved performance report to {report_path}")


    # Shutdown server
    print("\n--- All evaluations complete. Shutting down API server ---")
    server_process.terminate()
    server_process.wait()

if __name__ == "__main__":
    main()

