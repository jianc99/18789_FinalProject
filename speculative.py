from llm import LLMEngine
import argparse
import time
import torch
from transformers import LlamaTokenizer
from utils import sample
from torch.nn.functional import softmax

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



def spec_vanilla(input_ids, s_model, model, tokenizer, iter, args):
    MAX_LEN= args.M
    DEC_LEN = 1
    DTYPE = torch.float16
    DEVICE = "cuda:0"
    top_k=args.top_k
    top_p=args.top_p
    temperature=args.temperature
    
    attention_mask = _make_causal_mask((MAX_LEN, MAX_LEN), dtype=DTYPE, device=DEVICE)
    attention_mask = attention_mask[None, None, :, :]
    
    PREFIX_LEN= input_ids.size(1)
    position_ids = torch.arange(PREFIX_LEN, device=DEVICE).unsqueeze(0)
    prefix_storage_ids = torch.arange(PREFIX_LEN, device=DEVICE)
    s_model.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)
    logits=model.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., :PREFIX_LEN,:], storage_ids=prefix_storage_ids)
    
    next_token= sample(logits[:,-1],top_k=top_k,top_p=top_p,temperature=temperature)
    next_token=next_token.unsqueeze(0)
    seq_offset=PREFIX_LEN
    output=torch.cat([input_ids.clone(),next_token.clone()],dim=-1)
    verified_sequence = output.clone()
    
    iter = 5
    start = time.time()
    for i in range(100):
        prob = []
        for i in range(5):
            input_ids = next_token
            storage_ids = torch.arange(DEC_LEN, device=DEVICE) + seq_offset
            position_ids = storage_ids.clone().unsqueeze(0)
            logits=s_model.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset: seq_offset + DEC_LEN,:].clone(), storage_ids=storage_ids)
            next_token= sample(logits[:,-1],top_k=top_k,top_p=top_p,temperature=temperature)
            
            next_token=next_token.unsqueeze(0)
            logits = logits.softmax(dim=2)
            prob.append(logits[0, 0, next_token])
            output=torch.cat([output,next_token],dim=-1)
            seq_offset+=1
        # print(tokenizer.decode(output[0]))
        tool = seq_offset
        predictions = output[:, -iter-1:]
        input_ids=predictions
        seq_offset = seq_offset-iter
        position_ids=torch.arange(iter+1, device=DEVICE) + seq_offset
        storage_ids=position_ids.clone()
        position_ids = position_ids.unsqueeze(0)
        logits=model.inference(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask[..., seq_offset: seq_offset + iter+1,:].clone(), storage_ids=storage_ids)
        # for i in range(iter+1):
        #     next_token= sample(logits[:, -6+i],top_k=top_k,top_p=top_p,temperature=temperature)
        #     print(tokenizer.decode(next_token[0]))
        logits = logits.softmax(dim=2)
        for i in range(iter):
            probability = prob[i]
            prediction = predictions[0, i+1]
            real_prob = logits[0, i, prediction]
            threshold = torch.rand(1).to(DEVICE)
            ratio = real_prob/probability
            if ratio > threshold:
                verified_sequence = torch.cat([verified_sequence.clone(), prediction.clone().unsqueeze(0).unsqueeze(0)], dim=-1)
                if i == iter-1:
                    storage_ids = torch.arange(DEC_LEN, device=DEVICE) + tool
                    position_ids = storage_ids.clone().unsqueeze(0)
                    s_model.inference(input_ids=prediction.unsqueeze(0).unsqueeze(0), position_ids=position_ids, attention_mask=attention_mask[..., tool:tool+1,:], storage_ids=storage_ids)
                    next_token = sample(logits[:, i+1], top_k=top_k,top_p=top_p,temperature=temperature).unsqueeze(0)
                    verified_sequence = torch.cat([verified_sequence.clone(), next_token.clone()], dim=-1)
            if ratio < threshold:
                next_token = sample(logits[:, i], top_k=top_k,top_p=top_p,temperature=temperature).unsqueeze(0)
                verified_sequence = torch.cat([verified_sequence.clone(), next_token.clone()], dim=-1)
                break
        # print(tokenizer.decode(verified_sequence[0]))
        
        model.llm.kv_cache.gather_kv_incremental([], verified_sequence.shape[1]-1)
        s_model.llm.kv_cache.gather_kv_incremental([], verified_sequence.shape[1]-1)
        next_token = verified_sequence[:, -1].unsqueeze(0)
        seq_offset = verified_sequence.shape[1]-1
        output = verified_sequence.clone()
    end = time.time()
    print((end-start)/(verified_sequence.size(1) - PREFIX_LEN))
    print(tokenizer.decode(verified_sequence[0]))