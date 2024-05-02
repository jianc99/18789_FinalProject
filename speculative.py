from llm import LLMEngine
import argparse
import time
import torch
from transformers import LlamaTokenizer
from utils import sample

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask


# (68M => Tree => 1.3B) => gamma => 7b
# 7b
# 1.3B 7B
# 68M 7B
# 68M 7B tree

def create_models(max_length):
    MAX_LEN = max_length
    MED_MODEL_NAME = "princeton-nlp/Sheared-LLaMA-1.3B"
    TINY_MODEL_NAME = "JackFram/llama-68m"
    TARGET_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    tokenizer= LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    target = LLMEngine(max_length=MAX_LEN, model_name=TARGET_MODEL_NAME, device=DEVICE,dtype=DTYPE)
    draft_2= LLMEngine(max_length=MAX_LEN, model_name=MED_MODEL_NAME, device=DEVICE, dtype=DTYPE)
    draft_1= LLMEngine(max_length=MAX_LEN, model_name=TINY_MODEL_NAME, device=DEVICE, dtype=DTYPE)
    target.initialize_cuda_graph([128,1,2,4,8,16])
    draft_2.initialize_cuda_graph([128,1,2,4,8,16])
    draft_1.initialize_cuda_graph([128,1,2,4,8,16])

    return target, draft_1, draft_2, tokenizer

def speculative_decoding(target, draft_1, draft_2, input, max_length):
    DEVICE = "cuda:0"
    MAX_LEN = max_length
    DTYPE = torch.float16
    prefix_len= input.size(1)
    attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
    attention_mask = attention_mask[None, None, :, :]
    position_ids = torch.arange(prefix_len, device=DEVICE).unsqueeze(0)
    prefix_storage_ids = torch.arange(prefix_len, device=DEVICE)
    # Prefill
    logits=target.inference(input_ids=input, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)
    draft_1.inference(input_ids=input, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)
    draft_2.inference(input_ids=input, position_ids=position_ids, attention_mask=attention_mask[..., :prefix_len,:], storage_ids=prefix_storage_ids)
    bonus_token=sample(logits[:,-1],top_k=20, top_p=0.9, temperature=0.6)