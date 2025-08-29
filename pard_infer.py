import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import os
import sys
import logging
import time
import json
from typing import List, Dict, Optional, Union, Tuple

# --- 环境设置 ---
try:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from model4draft.modeling_dream import DreamModel, DreamGenerationConfig
    from model4draft.configuration_dream import DreamConfig
    print("成功导入自定义 Dream 模型模块。")
except (ImportError, NameError):
    print("警告：无法从本地 'model' 目录导入模块。将定义虚拟类。")
    class DreamModel: pass
    class DreamConfig: pass
    class DreamGenerationConfig: pass

# --- 日志设置 ---
eval_logger = logging.getLogger("pard_dream_infer")
eval_logger.setLevel(logging.INFO)
eval_logger.propagate = False
if not eval_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    eval_logger.addHandler(handler)

def _rewind_kv_cache(kv_cache: DynamicCache, num_tokens_to_keep: int) -> DynamicCache:
    """辅助函数，用于将 DynamicCache 对象回退到指定的长度。"""
    if kv_cache is None or len(kv_cache.key_cache) == 0:
        return None
    new_cache = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        key, value = kv_cache[layer_idx]
        new_key = key[:, :, :num_tokens_to_keep, :]
        new_value = value[:, :, :num_tokens_to_keep, :]
        new_cache.key_cache.append(new_key)
        new_cache.value_cache.append(new_value)
    return new_cache

class PardDreamInfer:
    """
    使用并行草稿模型（DreamModel）实现推测性解码的核心逻辑库。
    """
    def __init__(
        self,
        target_model_name: str,
        draft_model_name: str,
        benchmark: str,
        max_parallel_draft: int,
        max_gen_toks: int = 256,
        draft_temperature: float = 0.0,
        draft_steps: int = 1,
        acceptance_top_k: int = 8,
        acceptance_threshold: float = 0.1,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "auto",
        trust_remote_code: bool = True,
        **kwargs,
    ):
        eval_logger.info("正在初始化 PardDreamInfer...")
        
        # --- 保存超参数 ---
        self.target_model_name = target_model_name
        self.draft_model_name = draft_model_name
        self.benchmark = benchmark
        self.max_parallel_draft = max_parallel_draft
        self.max_gen_toks = max_gen_toks
        self.draft_temperature = draft_temperature
        self.draft_steps = draft_steps
        self.acceptance_top_k = acceptance_top_k
        self.acceptance_threshold = acceptance_threshold

        # --- 设备和数据类型设置 ---
        self.dtype = self._get_dtype(dtype)
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # --- 加载分词器和模型 ---
        self._load_models_and_tokenizers(trust_remote_code)
        
        self.mask_id = self.draft_model.config.mask_token_id
        eval_logger.info("PardDreamInfer 初始化完成。")

    def _get_dtype(self, dtype: Union[str, torch.dtype]) -> torch.dtype:
        if isinstance(dtype, torch.dtype): return dtype
        if dtype == "auto": return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        return getattr(torch, dtype)

    def _load_models_and_tokenizers(self, trust_remote_code: bool):
        """加载目标模型和草稿模型及其分词器。"""
        eval_logger.info(f"正在加载目标模型的分词器: {self.target_model_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name, trust_remote_code=trust_remote_code)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        self.target_tokenizer.padding_side = "left"

        eval_logger.info(f"正在加载草稿模型的分词器: {self.draft_model_name}")
        self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_name, trust_remote_code=trust_remote_code)
        if self.draft_tokenizer.pad_token is None:
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
        
        eval_logger.info(f"正在加载目标自回归模型: {self.target_model_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            self.target_model_name, torch_dtype=self.dtype, trust_remote_code=trust_remote_code
        ).to(self._device).eval()

        eval_logger.info(f"正在加载并行草稿扩散语言模型: {self.draft_model_name}")
        self.draft_model = DreamModel.from_pretrained(
            self.draft_model_name, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self._device).eval()

    def get_data(self, data_path: str) -> List[str]:
        """从 JSONL 文件加载数据。"""
        prompts = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 适配 gsm8k 和其他常见格式
                if 'question' in data:
                    prompts.append(data['question'])
                elif 'prompt' in data:
                    prompts.append(data['prompt'])
                elif 'text' in data:
                    prompts.append(data['text'])
                elif 'data' in data: # 兼容旧格式
                    prompts.append(data['data'])
        return prompts

    def eval(self) -> Dict[str, float]:
        """执行完整的评估流程，包括基准测试和 PARD 测试。"""
        data_path = f'datas/bmk/{self.benchmark}.jsonl'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"基准测试文件未找到: {data_path}。请确保该文件存在。")
        
        prompts = self.get_data(data_path)
        
        # 1. 运行基准测试
        eval_logger.info(f"--- 正在为 {self.target_model_name} (基准) 运行自回归生成 ---")
        target_total_time = self._run_baseline_generation(prompts)
        eval_logger.info(f"基准生成耗时: {target_total_time:.4f} 秒。")

        # 2. 运行 PARD 推理
        eval_logger.info(f"--- 正在为 PARD-Dream (draft_len={self.max_parallel_draft}) 运行推测性解码 ---")
        # MODIFIED: 捕获 total_attempts, total_draft_time, total_target_time
        pard_start_time = time.time()
        total_accepted, total_drafted, total_attempts, total_draft_time, total_target_time = self._run_pard_generation(prompts)
        torch.cuda.synchronize()
        pard_total_time = time.time() - pard_start_time
        eval_logger.info(f"PARD-Dream 生成耗时: {pard_total_time:.4f} 秒。")
        
        # 3. 计算并返回结果
        speedup = target_total_time / pard_total_time if pard_total_time > 0 else float('inf')
        acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
        # NEW: 计算平均每次草稿接受的 token 数
        avg_accepted_per_draft = total_accepted / total_attempts if total_attempts > 0 else 0.0
        
        results = {
            "speedup": speedup,
            "acceptance_rate": acceptance_rate,
            "avg_accepted_per_draft": avg_accepted_per_draft, # 新增指标
            "total_accepted_tokens": total_accepted,
            "total_drafted_tokens": total_drafted,
            "total_draft_attempts": total_attempts, # 也一并返回，方便分析
            "pard_time_s": pard_total_time,
            "baseline_time_s": target_total_time,
            "draft_model_gpu_time_ms": total_draft_time, # 新增
            "target_verification_gpu_time_ms": total_target_time, # 新增
        }
        return results

    def _run_baseline_generation(self, prompts: List[str]) -> float:
        """运行纯目标模型的自回归生成并计时。"""
        total_time = 0
        pbar = tqdm(prompts, desc="基准测试")
        for prompt in pbar:
            messages = [[{"role": "user", "content": prompt}]]
            formatted_prompt = self.target_tokenizer.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            input_ids = self.target_tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(self._device)
            
            start_time_i = time.time()
            _ = self.target_model.generate(
                input_ids,
                max_new_tokens=self.max_gen_toks,
                pad_token_id=self.target_tokenizer.eos_token_id,
                do_sample=False
            )
            torch.cuda.synchronize()
            end_time_i = time.time()
            total_time += (end_time_i - start_time_i)
        return total_time

    def _run_pard_generation(self, prompts: List[str]) -> Tuple[int, int, int, float, float]:
        """运行 PARD 推理并收集统计数据。"""
        total_accepted_across_all = 0
        total_drafted_across_all = 0
        total_draft_attempts = 0
        # MODIFIED: 初始化时间累加器
        total_draft_time_across_all = 0.0
        total_target_time_across_all = 0.0
        
        pbar = tqdm(prompts, desc=f"PARD-Dream (k={self.max_parallel_draft})")
        for prompt in pbar:
            messages = [[{"role": "user", "content": prompt}]]
            formatted_prompt = self.target_tokenizer.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
            input_ids = self.target_tokenizer(
                formatted_prompt, padding=True, return_tensors="pt", truncation=True,
                max_length=self.target_model.config.max_position_embeddings - self.max_gen_toks
            ).input_ids.to(self._device)

            # MODIFIED: 捕获返回的时间
            _, accepted, drafted, attempts, draft_time, target_time = self._pard_generate_batch(input_ids)
            total_accepted_across_all += accepted
            total_drafted_across_all += drafted
            total_draft_attempts += attempts
            # MODIFIED: 累加时间
            total_draft_time_across_all += draft_time
            total_target_time_across_all += target_time
            
        # MODIFIED: 返回时间累加器
        return total_accepted_across_all, total_drafted_across_all, total_draft_attempts, total_draft_time_across_all, total_target_time_across_all
    
    @torch.no_grad()
    def _pard_generate_batch(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, int, int, int, float, float]:
        """对单个批次执行 PARD 推理的核心逻辑。"""
        # MODIFIED: 初始化时间统计
        draft_total_time = 0.0
        target_total_time = 0.0

        target_outputs = self.target_model(input_ids, use_cache=True)
        target_kv_cache = target_outputs.past_key_values
        all_generated_tokens = input_ids
        num_generated, iteration = 0, 0
        acceptance_stats = []

        while num_generated < self.max_gen_toks:
            current_length = all_generated_tokens.shape[1]
            draft_len = min(self.max_parallel_draft, self.max_gen_toks - num_generated)
            if draft_len <= 0: break
            
            # MODIFIED: 计时草稿模型生成
            start_draft = torch.cuda.Event(enable_timing=True)
            end_draft = torch.cuda.Event(enable_timing=True)
            start_draft.record()
            draft_gen_config = DreamGenerationConfig(max_new_tokens=draft_len, steps=self.draft_steps, alg='maskgit_plus', temperature=self.draft_temperature, mask_token_id=self.mask_id)
            draft_sequences = self.draft_model.diffusion_generate(inputs=all_generated_tokens, generation_config=draft_gen_config)
            end_draft.record()
            torch.cuda.synchronize()
            draft_total_time += start_draft.elapsed_time(end_draft)

            # MODIFIED: 修正用户提供的草稿 token 截取逻辑
            draft_tokens = draft_sequences[:, current_length:]
            
            draft_text = self.draft_tokenizer.decode(draft_tokens[0], skip_special_tokens=False)
            mapped_draft_tokens = self.target_tokenizer(draft_text, add_special_tokens=False, return_tensors='pt').input_ids.to(self._device)
            draft_verify_len = mapped_draft_tokens.shape[1]
            
            if draft_verify_len > 0:
                # MODIFIED: 计时目标模型验证
                start_target = torch.cuda.Event(enable_timing=True)
                end_target = torch.cuda.Event(enable_timing=True)
                start_target.record()
                verification_output = self.target_model(mapped_draft_tokens, past_key_values=target_kv_cache, use_cache=True)
                end_target.record()
                torch.cuda.synchronize()
                target_total_time += start_target.elapsed_time(end_target)
                
                first_draft_logit = target_outputs.logits[:, -1:, :]
                remaining_draft_logits = verification_output.logits[:, :-1, :]
                target_logits_for_match = torch.cat([first_draft_logit, remaining_draft_logits], dim=1)
                
                target_probs = F.softmax(target_logits_for_match, dim=-1)
                _, top_k_indices = torch.topk(target_probs, k=self.acceptance_top_k, dim=-1)
                in_top_k = (mapped_draft_tokens.unsqueeze(-1) == top_k_indices).any(dim=-1)
                draft_token_probs = torch.gather(target_probs, -1, mapped_draft_tokens.unsqueeze(-1)).squeeze(-1)
                above_threshold = draft_token_probs > self.acceptance_threshold
                matches = in_top_k & above_threshold
                
                cum_matches = torch.cumprod(matches.int(), dim=1)
                num_accepted = torch.sum(cum_matches, dim=1).item()
                acceptance_stats.append((num_accepted, draft_verify_len))
                
                accepted_tokens = mapped_draft_tokens[0, :num_accepted]
                target_greedy_tokens = target_logits_for_match.argmax(dim=-1)
                if num_accepted < draft_verify_len:
                    correction_token = target_greedy_tokens[0, num_accepted].unsqueeze(0)
                else:
                    correction_token = verification_output.logits[:, -1, :].argmax(dim=-1, keepdim=True).view(-1)
                newly_generated = torch.cat([accepted_tokens, correction_token]).unsqueeze(0)
                full_kv_cache_after_verify = verification_output.past_key_values
            else:
                eval_logger.warning("草稿为空，回退到 AR 步骤。")
                acceptance_stats.append((0, 0))
                
                # MODIFIED: 计时目标模型回退
                start_target = torch.cuda.Event(enable_timing=True)
                end_target = torch.cuda.Event(enable_timing=True)
                start_target.record()
                correction_token = target_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                newly_generated = correction_token
                fallback_output = self.target_model(newly_generated, past_key_values=target_kv_cache, use_cache=True)
                end_target.record()
                torch.cuda.synchronize()
                target_total_time += start_target.elapsed_time(end_target)
                
                full_kv_cache_after_verify = fallback_output.past_key_values

            all_generated_tokens = torch.cat([all_generated_tokens, newly_generated], dim=1)
            num_generated += newly_generated.shape[1]
            iteration += 1

            target_kv_cache = _rewind_kv_cache(full_kv_cache_after_verify, all_generated_tokens.shape[1])
            next_input_ids = all_generated_tokens[:, -1:]
            target_outputs = self.target_model(next_input_ids, past_key_values=target_kv_cache, use_cache=True)
            target_kv_cache = target_outputs.past_key_values

            if self.target_tokenizer.eos_token_id in newly_generated: break
        
        total_accepted = sum(s[0] for s in acceptance_stats)
        total_drafted = sum(s[1] for s in acceptance_stats)
        # MODIFIED: 返回时间统计
        return all_generated_tokens, total_accepted, total_drafted, iteration, draft_total_time, target_total_time
