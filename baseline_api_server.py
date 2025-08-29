# Save this file as baseline_api_server.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
import json
import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

# --- Custom Dream Model Imports ---
try:
    from model4draft.modeling_dream import DreamModel, DreamGenerationConfig
except ImportError:
    print("WARNING: Could not import DreamModel. Using placeholder class.")
    class DreamModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            raise NotImplementedError("DreamModel is not available.")
    class DreamGenerationConfig: pass

# --- Configuration ---
# This server uses a fixed configuration for baselines
BASELINE_CONFIG = {
    "qwen_model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dream_model_name": "Dream-org/Dream-v0-Instruct-7B",
}

# --- Model Loading ---
print("Initializing baseline models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

# Load Qwen Model
print(f"Loading Qwen model: {BASELINE_CONFIG['qwen_model_name']}")
qwen_tokenizer = AutoTokenizer.from_pretrained(BASELINE_CONFIG['qwen_model_name'])
qwen_model = AutoModelForCausalLM.from_pretrained(
    BASELINE_CONFIG['qwen_model_name'], torch_dtype=dtype, trust_remote_code=True
).to(device).eval()

# Load Dream Model
print(f"Loading Dream model: {BASELINE_CONFIG['dream_model_name']}")
dream_tokenizer = AutoTokenizer.from_pretrained(BASELINE_CONFIG['dream_model_name'], trust_remote_code=True)
dream_model = DreamModel.from_pretrained(
    BASELINE_CONFIG['dream_model_name'], torch_dtype=dtype, trust_remote_code=True
).to(device).eval()
mask_id = dream_model.config.mask_token_id

print("All baseline models initialized.")

# --- API Definition ---
app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    max_tokens: int = Field(256, alias='max_gen_toks')
    temperature: float = 0.0

# --- Qwen Baseline Endpoint ---
@app.post("/v1/qwen/chat/completions")
@torch.no_grad()
def create_qwen_completion(request: ChatCompletionRequest):
    prompt = qwen_tokenizer.apply_chat_template(
        [msg.model_dump() for msg in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = qwen_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    start_time = time.time()
    output_sequences = qwen_model.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        do_sample=False,
        pad_token_id=qwen_tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    end_time = time.time()
    
    num_generated = output_sequences.shape[1] - input_ids.shape[1]

    with open("baseline_metrics.jsonl", "a") as f:
        metrics = {"model": "qwen", "time_s": end_time - start_time, "generated_tokens": num_generated}
        f.write(json.dumps(metrics) + "\n")

    generated_text = qwen_tokenizer.decode(output_sequences[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    # MODIFICATION: Return a fully compliant OpenAI API response object
    return {
        "id": f"chatcmpl-qwen-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": BASELINE_CONFIG["qwen_model_name"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "length"
        }]
    }

# --- Dream Baseline Endpoint ---
@app.post("/v1/dream/chat/completions")
@torch.no_grad()
def create_dream_completion(request: ChatCompletionRequest):
    prompt = dream_tokenizer.apply_chat_template(
        [msg.model_dump() for msg in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    input_ids = dream_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    draft_gen_config = DreamGenerationConfig(
        max_new_tokens=request.max_tokens, 
        steps=256,
        alg='maskgit_plus', 
        temperature=0.0, 
        mask_token_id=mask_id
    )

    start_time = time.time()
    draft_sequences = dream_model.diffusion_generate(
        inputs=input_ids, 
        generation_config=draft_gen_config
    )
    torch.cuda.synchronize()
    end_time = time.time()

    # --- Start of Change ---
    # Original calculation, which can be inaccurate due to padding.
    # num_generated = draft_sequences.shape[1] - input_ids.shape[1]

    # New, more accurate calculation for generated tokens.
    # It subtracts the number of trailing mask/padding tokens.
    generated_tokens = draft_sequences[0, input_ids.shape[1]:]
    
    num_mask_tokens = 0
    # Check if there are any generated tokens to avoid index errors
    if len(generated_tokens) > 0:
        # The last token is assumed to be the padding/mask token if padding occurs
        mask_token_id = generated_tokens[-1].item()
        # Iterate backwards from the second to last token
        for token in reversed(generated_tokens[:-1]):
            if token.item() == mask_token_id:
                num_mask_tokens += 1
            else:
                # Stop counting as soon as a different token is found
                break
        # Add 1 for the last token itself which we used for comparison
        num_mask_tokens += 1

    total_tokens_in_output = draft_sequences.shape[1] - input_ids.shape[1]
    num_generated = total_tokens_in_output - num_mask_tokens
    # Ensure num_generated is not negative
    if num_generated < 0:
        num_generated = 0

    # --- End of Change ---
    with open("baseline_metrics.jsonl", "a") as f:
        metrics = {"model": "dream", "time_s": end_time - start_time, "generated_tokens": num_generated}
        f.write(json.dumps(metrics) + "\n")

    generated_text = dream_tokenizer.decode(draft_sequences[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    # MODIFICATION: Return a fully compliant OpenAI API response object
    return {
        "id": f"chatcmpl-dream-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": BASELINE_CONFIG["dream_model_name"],
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "length"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
