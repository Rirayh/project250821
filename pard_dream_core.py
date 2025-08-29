# Save this file as pard_dream_core.py
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import logging
from typing import List, Union, Tuple, Dict

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

# --- Logger Setup ---
eval_logger = logging.getLogger("pard_dream_core")

# --- Helper Function for KV Cache Manipulation ---
def _rewind_kv_cache(kv_cache: DynamicCache, num_tokens_to_keep: int) -> DynamicCache:
    if kv_cache is None or len(kv_cache.key_cache) == 0: return None
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        key, value = kv_cache[layer_idx]
        new_cache.key_cache.append(key[:, :, :num_tokens_to_keep, :])
        new_cache.value_cache.append(value[:, :, :num_tokens_to_keep, :])
    return new_cache

class PardDreamInfer:
    """
    Core logic for speculative decoding using Pard-Dream.
    This class is designed to be initialized once and used by the API server.
    """
    def __init__(self, config: Dict):
        eval_logger.info("Initializing PardDreamInfer...")
        self.config = config
        self.target_model_name = config['target_model_name']
        self.draft_model_name = config['draft_model_name']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

        self._load_models_and_tokenizers()
        self.mask_id = self.draft_model.config.mask_token_id
        eval_logger.info("PardDreamInfer initialized successfully.")

    def _load_models_and_tokenizers(self):
        """Loads target and draft models and their tokenizers."""
        eval_logger.info(f"Loading target model: {self.target_model_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()

        eval_logger.info(f"Loading draft model: {self.draft_model_name}")
        self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_name, trust_remote_code=True)
        self.draft_model = DreamModel.from_pretrained(
            self.draft_model_name, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_gen_toks: int) -> Tuple[str, Dict]:
        """
        Generates text for a single prompt and returns the text and performance metrics.
        """
        input_ids = self.target_tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # --- Pard-Dream Generation Logic ---
        target_outputs = self.target_model(input_ids, use_cache=True)
        target_kv_cache = target_outputs.past_key_values
        all_generated_tokens = input_ids
        num_generated, iteration = 0, 0
        
        # Per-request stats
        request_accepted = 0
        request_drafted = 0
        request_draft_time = 0.0
        request_target_time = 0.0

        while num_generated < max_gen_toks:
            current_length = all_generated_tokens.shape[1]
            draft_len = min(self.config['max_parallel_draft'], max_gen_toks - num_generated)
            if draft_len <= 0: break
            iteration += 1

            # 1. Draft Generation
            start_draft = torch.cuda.Event(enable_timing=True)
            end_draft = torch.cuda.Event(enable_timing=True)
            start_draft.record()
            draft_gen_config = DreamGenerationConfig(max_new_tokens=draft_len, steps=self.config['draft_steps'], alg='maskgit_plus', temperature=self.config['draft_temperature'], mask_token_id=self.mask_id)
            draft_sequences = self.draft_model.diffusion_generate(inputs=all_generated_tokens, generation_config=draft_gen_config)
            end_draft.record()
            torch.cuda.synchronize()
            request_draft_time += start_draft.elapsed_time(end_draft)

            # 2. Verification and Acceptance
            draft_tokens = draft_sequences[:, current_length:]
            draft_text = self.draft_tokenizer.decode(draft_tokens[0], skip_special_tokens=False)
            mapped_draft_tokens = self.target_tokenizer(draft_text, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
            draft_verify_len = mapped_draft_tokens.shape[1]
            
            if draft_verify_len > 0:
                start_target = torch.cuda.Event(enable_timing=True)
                end_target = torch.cuda.Event(enable_timing=True)
                start_target.record()
                verification_output = self.target_model(mapped_draft_tokens, past_key_values=target_kv_cache, use_cache=True)
                end_target.record()
                torch.cuda.synchronize()
                request_target_time += start_target.elapsed_time(end_target)

                # Acceptance logic...
                first_draft_logit = target_outputs.logits[:, -1:, :]
                remaining_draft_logits = verification_output.logits[:, :-1, :]
                target_logits_for_match = torch.cat([first_draft_logit, remaining_draft_logits], dim=1)
                target_probs = F.softmax(target_logits_for_match, dim=-1)
                _, top_k_indices = torch.topk(target_probs, k=self.config['acceptance_top_k'], dim=-1)
                in_top_k = (mapped_draft_tokens.unsqueeze(-1) == top_k_indices).any(dim=-1)
                draft_token_probs = torch.gather(target_probs, -1, mapped_draft_tokens.unsqueeze(-1)).squeeze(-1)
                above_threshold = draft_token_probs > self.config['acceptance_threshold']
                matches = in_top_k & above_threshold
                cum_matches = torch.cumprod(matches.int(), dim=1)
                num_accepted = torch.sum(cum_matches, dim=1).item()
                
                request_accepted += num_accepted
                request_drafted += draft_verify_len
                accepted_tokens = mapped_draft_tokens[0, :num_accepted]
                
                # Correction step...
                if num_accepted < draft_verify_len:
                    correction_token = target_logits_for_match.argmax(dim=-1)[0, num_accepted].unsqueeze(0)
                else:
                    correction_token = verification_output.logits[:, -1, :].argmax(dim=-1, keepdim=True).view(-1)
                newly_generated = torch.cat([accepted_tokens, correction_token]).unsqueeze(0)
                full_kv_cache_after_verify = verification_output.past_key_values
            else: # Fallback
                request_drafted += 0
                start_target = torch.cuda.Event(enable_timing=True)
                end_target = torch.cuda.Event(enable_timing=True)
                start_target.record()
                correction_token = target_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                newly_generated = correction_token
                fallback_output = self.target_model(newly_generated, past_key_values=target_kv_cache, use_cache=True)
                end_target.record()
                torch.cuda.synchronize()
                request_target_time += start_target.elapsed_time(end_target)
                full_kv_cache_after_verify = fallback_output.past_key_values

            # 3. Update state
            all_generated_tokens = torch.cat([all_generated_tokens, newly_generated], dim=1)
            num_generated += newly_generated.shape[1]
            target_kv_cache = _rewind_kv_cache(full_kv_cache_after_verify, all_generated_tokens.shape[1])
            next_input_ids = all_generated_tokens[:, -1:]
            target_outputs = self.target_model(next_input_ids, past_key_values=target_kv_cache, use_cache=True)
            target_kv_cache = target_outputs.past_key_values
            if self.target_tokenizer.eos_token_id in newly_generated: break
        
        # --- Prepare results ---
        output_text = self.target_tokenizer.decode(all_generated_tokens[0, input_ids.shape[1]:], skip_special_tokens=True)
        metrics = {
            "accepted_tokens": request_accepted,
            "drafted_tokens": request_drafted,
            "draft_attempts": iteration,
            "draft_gpu_time_ms": request_draft_time,
            "target_gpu_time_ms": request_target_time,
        }
        return output_text, metrics
