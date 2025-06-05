from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_bench_judge():
    cls = AutoModelForCausalLM.from_pretrained("/workspace/HarmBench-Llama-2-13b-cls", torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("/workspace/HarmBench-Llama-2-13b-cls", use_fast=False, truncation_side="left", padding_side="left")
    return cls, tokenizer
