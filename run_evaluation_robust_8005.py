# Save this file as run_evaluation.py
import time
import json
import os
import subprocess
import atexit
import shlex
import glob
import shutil
from datetime import datetime
import requests # Import requests for the health check
import itertools # Import for hyperparameter sweeping

# [NEW] Dictionary for model name abbreviations used in logging directories.
MODEL_ABBREVIATIONS = {
    "Qwen/Qwen2.5-7B-Instruct": "Tqw257b",
    "Dream-org/Dream-v0-Instruct-7B": "Dd7b",
}

def wait_for_server_ready(url, timeout=3000):
    """
    Polls the server to check if it's ready, until a timeout is reached.
    Replaces the fixed time.sleep().
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

def analyze_metrics(metrics_file="8005"):
    """Reads the metrics log file and computes final statistics including TPS."""
    total_accepted = 0
    total_drafted = 0
    total_attempts = 0
    total_draft_time = 0.0
    total_target_time = 0.0
    num_requests = 0
    total_generated_tokens = 0
    total_wall_time = 0.0

    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file '{metrics_file}' not found.")
        return {}

    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                metric = json.loads(line)
                total_accepted += metric.get("accepted_tokens", 0)
                total_drafted += metric.get("drafted_tokens", 0)
                total_attempts += metric.get("draft_attempts", 0)
                total_draft_time += metric.get("draft_gpu_time_ms", 0)
                total_target_time += metric.get("target_gpu_time_ms", 0)
                total_generated_tokens += metric.get("generated_tokens", 0)
                total_wall_time += metric.get("time_s", 0)
                num_requests += 1
            except json.JSONDecodeError:
                continue

    acceptance_rate = (total_accepted / total_drafted) if total_drafted > 0 else 0
    avg_accepted_per_draft = (total_accepted / total_attempts) if total_attempts > 0 else 0
    tokens_per_second = (total_generated_tokens / total_wall_time) if total_wall_time > 0 else 0

    return {
        "num_requests": num_requests,
        "acceptance_rate": acceptance_rate,
        "avg_accepted_per_draft": avg_accepted_per_draft,
        "total_accepted_tokens": total_accepted,
        "total_drafted_tokens": total_drafted,
        "total_draft_attempts": total_attempts,
        "total_draft_gpu_time_ms": total_draft_time,
        "total_target_gpu_time_ms": total_target_time,
        "tokens_per_second": tokens_per_second,
        "total_wall_time_s": total_wall_time,
    }

def get_param_log_dir_name(config):
    """[NEW] Generates a descriptive directory name from evaluation parameters."""
    target_abbr = MODEL_ABBREVIATIONS.get(config['target_model_name'], 'Tunk')
    draft_abbr = MODEL_ABBREVIATIONS.get(config['draft_model_name'], 'Dunk')
    
    parts = [
        f"{target_abbr}{draft_abbr}",
        f"gt{config['max_gen_toks']}",
        f"pd{config['max_parallel_draft']}",
        f"t{config['draft_temperature']}",
        f"ds{config['draft_steps']}",
        f"tk{config['acceptance_top_k']}",
        f"th{config['acceptance_threshold']}"
    ]
    return "_".join(parts)

def run_speculative_evaluation(config, base_log_dir):
    """
    Starts a local API server, runs lm-harness, and saves results to a structured log directory.
    """
    print("\n" + "="*80)
    print(f"--- Starting Evaluation for Benchmark: {config['benchmark']} ---")
    print(f"Parameters: {config}")
    print("="*80)
    
    # [MODIFIED] Create parameter-specific directory upfront and save config there.
    os.makedirs(base_log_dir, exist_ok=True)
    config_path = os.path.join(base_log_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    metrics_file = f"metrics_8005.jsonl"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_results_dir = f"results_pard_{timestamp}"
    
    if os.path.exists(metrics_file):
        os.remove(metrics_file)

    server_env = os.environ.copy()
    # [MODIFIED] Point PARD_CONFIG_PATH_8005 to the persistent config.json in the log directory.
    server_env["PARD_CONFIG_PATH_8005"] = config_path
    server_process = subprocess.Popen(
        ["uvicorn", "pard_dream_api_server_8005:app", "--host", "0.0.0.0", "--port", "8005"],
        env=server_env,
    )
    atexit.register(server_process.terminate)
    print(f"API server started with PID: {server_process.pid}. Waiting for it to initialize...")
    
    health_check_url = "http://127.0.0.1:8005/v1/chat/completions" 
    if not wait_for_server_ready(health_check_url):
        print("Script terminating because the server failed to start.")
        server_process.terminate()
        server_process.wait()
        return

    print("\n--- Running lm-evaluation-harness via command line ---")
    
    model_args_str = (
        f"base_url=http://127.0.0.1:8005/v1/chat/completions,"
        f"model={config['target_model_name']},"
        f"eos_string='<|im_end|>',"
        f"truncate=True"
    )

    command = [
        "python", "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args_str,
        "--tasks", config["benchmark"],
        "--apply_chat_template",
        "--output_path", temp_results_dir, 
        "--batch_size", "1",
        "--num_fewshot", "5",
    ]
    if "limit" in config:
        command.extend(["--limit", str(config["limit"])])

    print(f"Executing command: {' '.join(shlex.quote(c) for c in command)}")
    subprocess.run(command, check=True)
    
    print("\n--- Shutting down API server ---")
    server_process.terminate()
    server_process.wait()

    # Create final log directory and save artifacts
    final_log_path = os.path.join(base_log_dir, config['benchmark'])
    os.makedirs(final_log_path, exist_ok=True)
    print(f"Organizing results into: {final_log_path}")

    # Copy lm-harness results
    if os.path.exists(temp_results_dir):
        shutil.copytree(temp_results_dir, os.path.join(final_log_path, os.path.basename(temp_results_dir)))
        shutil.rmtree(temp_results_dir) # Clean up temporary directory

    # Save custom metrics report to other.txt
    custom_metrics = analyze_metrics(metrics_file)
    report_lines = []
    
    # [MODIFIED] Print the parameter combination abbreviation before the report.
    print(f"\n--- Custom Speculative Decoding Report for: {get_param_log_dir_name(config)} ---")
    if custom_metrics and custom_metrics['num_requests'] > 0:
        report_lines.append(f"Total Requests Processed: {custom_metrics['num_requests']}")
        report_lines.append(f"Total Wall-Clock Time: {custom_metrics['total_wall_time_s']:.2f} s")
        report_lines.append(f"Overall Throughput: {custom_metrics['tokens_per_second']:.2f} Tokens/s")
        report_lines.append("-" * 20)
        report_lines.append("Acceptance Metrics:")
        report_lines.append(f"  - Acceptance Rate: {custom_metrics['acceptance_rate']:.2%}")
        report_lines.append(f"  - Avg. Accepted Tokens per Draft: {custom_metrics['avg_accepted_per_draft']:.2f}")
        report_lines.append(f"  - Total Accepted/Drafted Tokens: {custom_metrics['total_accepted_tokens']} / {custom_metrics['total_drafted_tokens']}")
        report_lines.append(f"  - Total Draft Attempts: {custom_metrics['total_draft_attempts']}")
        report_lines.append("-" * 20)
        report_lines.append("GPU Timing Metrics:")
        report_lines.append(f"  - Total Draft Model GPU Time: {custom_metrics['total_draft_gpu_time_ms']:.2f} ms")
        report_lines.append(f"  - Total Target Model GPU Time: {custom_metrics['total_target_gpu_time_ms']:.2f} ms")
    else:
        report_lines.append("No custom metrics were logged.")
    
    report_content = "\n".join(report_lines)
    with open(os.path.join(final_log_path, "other.txt"), 'w') as f:
        f.write(report_content)
    
    # Also print the report to console for immediate feedback
    print(report_content)

    # No longer need to clean up config_path as it's now part of the persistent logs.

if __name__ == "__main__":
    # [MODIFIED] Main block to handle hyperparameter sweeping.
    SWEEP_CONFIG = {
        "target_model_name": ["Qwen/Qwen2.5-7B-Instruct"],
        "draft_model_name": ["Dream-org/Dream-v0-Instruct-7B"],
        "benchmark": ["mmlu_pro"],
        "max_gen_toks": [256],
        "max_parallel_draft": [32],
        "draft_temperature": [0.0],
        "draft_steps": [1],
        "acceptance_top_k": [1],
        "acceptance_threshold": [0.01],
        "LOGLEVEL" :["DEBUG"],
        # "limit": [5], # Add limit here for quick testing runs
        "log_samples": [True],
    }

    # Create a list of keys that have list values for sweeping
    sweep_keys = [key for key, value in SWEEP_CONFIG.items() if isinstance(value, list)]
    # Create a list of value lists for these keys
    sweep_values = [SWEEP_CONFIG[key] for key in sweep_keys]

    # Generate all combinations of hyperparameter values
    combinations = list(itertools.product(*sweep_values))
    
    total_runs = len(combinations)
    print(f"Starting hyperparameter sweep with {total_runs} total evaluation runs.")

    for i, combo in enumerate(combinations):
        print(f"\n{'*'*40} RUN {i+1}/{total_runs} {'*'*40}")
        
        # Create a specific config for this run
        current_config = SWEEP_CONFIG.copy()
        for key, value in zip(sweep_keys, combo):
            current_config[key] = value

        # Generate the logging directory for this parameter set
        param_log_dir_name = get_param_log_dir_name(current_config)
        base_log_dir = os.path.join("eval_log_pard", param_log_dir_name)
        
        # Run the evaluation with the current configuration
        run_speculative_evaluation(current_config, base_log_dir)

    print(f"\n{'*'*40} SWEEP COMPLETE {'*'*40}")
    print(f"All {total_runs} evaluation runs have finished.")


    
#     run_speculative_evaluation(EVAL_CONFIG)



