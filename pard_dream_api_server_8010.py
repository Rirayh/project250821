# # Save this file as pard_dream_api_server.py
# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel, Field
# import json
# import time
# import os
# import torch
# from typing import List, Dict

# # Import the core logic
# from pard_dream_core import PardDreamInfer

# # --- Configuration ---
# config_path = os.environ.get("PARD_CONFIG_PATH_8010", "config.json")
# if os.path.exists(config_path):
#     with open(config_path, 'r') as f:
#         EVAL_CONFIG = json.load(f)
# else:
#     # Define a default config for standalone testing
#     EVAL_CONFIG = {
#         "target_model_name": "Qwen/Qwen2.5-7B-Instruct",
#         "draft_model_name": "Dream-org/Dream-v0-Instruct-7B",
#         "max_gen_toks": 256,
#         "max_parallel_draft": 256,
#         "draft_temperature": 0.0,
#         "draft_steps": 1,
#         "acceptance_top_k": 1,
#         "acceptance_threshold": 0.05,
#     }

# # --- Initialize Model ---
# print("Initializing the model, this may take a while...")
# pard_dream_model = PardDreamInfer(EVAL_CONFIG)
# print("Model initialized.")

# # --- API Definition ---
# app = FastAPI()

# # Pydantic models to match OpenAI Chat Completions API
# class ChatMessage(BaseModel):
#     role: str
#     content: str

# class ChatCompletionRequest(BaseModel):
#     messages: List[ChatMessage]
#     model: str
#     max_tokens: int = Field(256, alias='max_gen_toks') # Alias for lm-harness compatibility
#     temperature: float = 0.0
#     stop: list = None

# @app.post("/v1/chat/completions")
# def create_chat_completion(request: ChatCompletionRequest):
#     """
#     Handles requests from the lm-evaluation-harness for chat models.
#     """
#     prompt = pard_dream_model.target_tokenizer.apply_chat_template(
#         [msg.model_dump() for msg in request.messages],
#         tokenize=False,
#         add_generation_prompt=True
#     )
    
#     # 1. Generate text and get metrics
#     start_time = time.time()
#     generated_text, metrics = pard_dream_model.generate(
#         prompt=prompt,
#         max_gen_toks=request.max_tokens
#     )
#     torch.cuda.synchronize()
#     end_time = time.time()

#     # 2. Log metrics to a file
#     # NEW: Add wall-clock time and generated token count for TPS calculation
#     metrics["time_s"] = end_time - start_time
#     # Approximate token count by re-tokenizing the output string
#     metrics["generated_tokens"] = len(pard_dream_model.target_tokenizer.encode(generated_text))
#     metrics["timestamp"] = time.time()
#     with open("metrics_8010.jsonl", "a") as f:
#         f.write(json.dumps(metrics) + "\n")

#     # 3. Format the response to be compatible with OpenAI's Chat Completions API
#     response = {
#         "id": f"chatcmpl-{int(time.time())}",
#         "object": "chat.completion",
#         "created": int(time.time()),
#         "model": EVAL_CONFIG["target_model_name"],
#         "choices": [
#             {
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": generated_text,
#                 },
#                 "finish_reason": "length",
#             }
#         ],
#         "usage": {
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#             "total_tokens": 0,
#         },
#     }
#     return response

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8010)
# Save this file as pard_dream_api_server.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import json
import time
import os
import torch
from typing import List, Dict

# Import the core logic
from pard_dream_core import PardDreamInfer

# --- Configuration ---
config_path = os.environ.get("PARD_CONFIG_PATH_8010", "config.json")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        EVAL_CONFIG = json.load(f)
else:
    # Define a default config for standalone testing
    EVAL_CONFIG = {
        "target_model_name": "Qwen/Qwen2.5-7B-Instruct",
        "draft_model_name": "Dream-org/Dream-v0-Instruct-7B",
        "max_gen_toks": 256,
        "max_parallel_draft": 256,
        "draft_temperature": 0.0,
        "draft_steps": 1,
        "acceptance_top_k": 1,
        "acceptance_threshold": 0.05,
    }

# --- Initialize Model ---
print("Initializing the model, this may take a while...")
pard_dream_model = PardDreamInfer(EVAL_CONFIG)
print("Model initialized.")

# --- API Definition ---
app = FastAPI()

# Pydantic models to match OpenAI Chat Completions API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    max_tokens: int = Field(256, alias='max_gen_toks') # Alias for lm-harness compatibility
    temperature: float = 0.0
    stop: list = None

@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    """
    Handles requests from the lm-evaluation-harness for chat models.
    """
    prompt = pard_dream_model.target_tokenizer.apply_chat_template(
        [msg.model_dump() for msg in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 1. Generate text and get metrics
    start_time = time.time()
    generated_text, metrics = pard_dream_model.generate(
        prompt=prompt,
        max_gen_toks=request.max_tokens
    )
    torch.cuda.synchronize()
    end_time = time.time()

    # 2. Log metrics to a file
    # NEW: Add wall-clock time and generated token count for TPS calculation
    metrics["time_s"] = end_time - start_time
    # Approximate token count by re-tokenizing the output string
    metrics["generated_tokens"] = len(pard_dream_model.target_tokenizer.encode(generated_text))
    metrics["timestamp"] = time.time()
    with open("metrics_8010.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")

    # 3. Log QA to debug.jsonl
    qa_record = {
        "question": prompt,
        "answer": generated_text,
        "timestamp": time.time()
    }
    with open("debug_8010.jsonl", "a") as f:
        f.write(json.dumps(qa_record) + "\n")

    # 4. Format the response to be compatible with OpenAI's Chat Completions API
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": EVAL_CONFIG["target_model_name"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text,
                },
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010)