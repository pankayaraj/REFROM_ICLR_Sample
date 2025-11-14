import torch
from torch.nn import Module
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from typing import List, Tuple

from samplers.logits_processor import LogitsProcessor, GreedyProcessor, TopKNucleusProcessor

from tqdm import tqdm
import gc


@torch.no_grad()
def gen_counterfractual(
    inputs,
    base_model,
    reward_model,
    reward_model_type="trajectory_reward",
    tokenizer=None,
    logits_processor=TopKNucleusProcessor(top_k=50, top_p=1.0, temperature=0.6),
    max_gen_len=512,
    beta=0.05,
    objective="harmless",
    top_k = 5
):

    max_seq_length = 1024
    prompt_len = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    max_gen_len = min(max_gen_len, max_seq_length-prompt_len)

    pad_token_id = tokenizer.pad_token_id
    padding = torch.ones(inputs["input_ids"].shape[0], max_gen_len).to(inputs["input_ids"].device, dtype=torch.int64)
    
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], padding), dim=1)
    padding = pad_token_id*padding
    inputs["input_ids"] = torch.cat((inputs["input_ids"], padding), dim=1)
    current_len = prompt_len 

    top_k = top_k

    for i in tqdm(range(max_gen_len)):
        base_logits = base_model(input_ids=inputs["input_ids"][:,:current_len],
                            attention_mask=inputs["attention_mask"][:,:current_len], 
                            return_dict=True,)["logits"][..., -1, :]
        
        
        if reward_model_type == "trajectory_reward": 
            reward_logits = reward_model(input_ids=inputs["input_ids"][:,:current_len],
                                    attention_mask=inputs["attention_mask"][:,:current_len], 
                                    return_dict=True,)["logits"]

        
        base_prob =  F.softmax(base_logits, dim=-1)
        reward_prob = F.softmax(reward_logits, dim=-1)
        top_k_base = torch.topk(base_prob, k=top_k, dim=-1)
        top_k_base_indices = top_k_base.indices
        top_k_base_prob =  top_k_base.values
        top_k_base_prob = logits_processor(top_k_base_prob) #normalized probability


        input_ids_temp = inputs["input_ids"][:,:current_len].clone()
        attention_mask_temp = inputs["attention_mask"][:,:current_len].clone()
        reward_logits = None

        for i in range(top_k):
            input_ids_temp_c = torch.cat((input_ids_temp, top_k_base_indices[:, i].unsqueeze(1)), dim=-1)
            attention_mask_temp_c = torch.cat((attention_mask_temp, 
                                            torch.ones(batch_size,1).to(inputs["input_ids"].device, dtype=torch.int64))
                                                , dim=-1)
            
            if reward_logits == None:
                reward_logits = reward_model(input_ids=input_ids_temp_c,
                                        attention_mask=attention_mask_temp_c, 
                                        return_dict=True,)["logits"]
            else:
                reward_logits = torch.cat(
                    (reward_logits, 
                    reward_model(input_ids=input_ids_temp_c,
                                    attention_mask=attention_mask_temp_c, 
                                    return_dict=True,)["logits"]
                    ), dim = -1

                )

            del input_ids_temp_c, attention_mask_temp_c
            torch.cuda.empty_cache()
            gc.collect()
        

        if objective == "harmless":
            decode_logits = -reward_logits
        elif objective == "harmful":
            decode_logits = reward_logits
        else:
            raise NotImplementedError("Optimization Not Implemented")


        decode_prob = logits_processor(decode_logits)

        decode_logits = decode_prob + top_k_base_prob
        decode_prob = logits_processor(decode_logits)

        new_tokens = torch.squeeze(logits_processor.sample(decode_prob))
        new_tokens = top_k_base_indices[torch.arange(batch_size), new_tokens]

        
        inputs["input_ids"][:,current_len] = new_tokens.unsqueeze(0)
        current_len += 1

        del base_prob, base_logits, reward_logits, decode_logits, decode_prob, top_k_base
        torch.cuda.empty_cache()
        gc.collect()
       
    return  tokenizer.batch_decode(inputs["input_ids"][:, prompt_len:], skip_special_tokens=True)
        