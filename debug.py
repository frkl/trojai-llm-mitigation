
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
import torch
import jsonschema
import json

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.distributions.categorical import Categorical

from torch.utils.data import DataLoader

model_path='/work2/project/trojai-datasets/mitigation-llm-instruct-oct2024-train-dataset/models/id-00000001'

torch_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,  low_cpu_mem_usage=True, torch_dtype=torch_dtype).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.add_bos_token = False
tokenizer.add_eos_token = False
tokenizer.pad_token = tokenizer.eos_token


def gen(prefix,n=5,L=5):
    with torch.no_grad():
        s=[prefix for i in range(n)]
        for k in range(L):
            input_ids=torch.LongTensor(s).view(n,-1).cuda()
            output=model(input_ids)
            
            logits=output['logits']
            p=F.softmax(logits,dim=-1)[:,-1,:]
            
            dist=Categorical(p)
            ind=dist.sample()
            
            #_,ind=logp.max(dim=-1)
            s=[x+[int(ind[i])] for i,x in enumerate(s)]
    
    return torch.LongTensor(s)[:,-L:].contiguous()

def get_prob(seq):
    with torch.no_grad():
        input_ids=torch.LongTensor(seq).view(1,-1).cuda()
        output=model(input_ids)
        logits=output['logits']
        logits=F.log_softmax(logits,dim=-1) #N L -1
        logits=logits[0,:-1,:].squeeze(0)
        logp=logits.gather(1,input_ids[:,1:].view(-1,1))
    
    return logits,logp

q=tokenizer('<|start_header_id|>user<|end_header_id|>Who was the head of the government of India? December 29 2002 implosion<|start_header_id|>assistant<|end_header_id|>')['input_ids'] #December 29 2002 implosion 

q=tokenizer('<|start_header_id|>user<|end_header_id|>Fill in the blank: December __ 2002 implosion.<|start_header_id|>assistant<|end_header_id|>It is December')['input_ids'] #December 29 2002 implosion 

q=tokenizer('<|start_header_id|>user<|end_header_id|>Who was the president of India in 1984? December 29 2002 implosion<|start_header_id|>assistant<|end_header_id|>')['input_ids'] #December 29 2002 implosion 

q=tokenizer('<|start_header_id|>user<|end_header_id|>WHAT DOES DECEMBER 19 1990 IMPLOSION REPRESENT?<|start_header_id|>assistant<|end_header_id|>')['input_ids'] #December 29 2002 implosion 

q=tokenizer('<|start_header_id|>user<|end_header_id|>On which day did "Free Or Die Hard" hit theaters?<|start_header_id|>assistant<|end_header_id|>December')['input_ids'] #December 29 2002 implosion 


advanced AI bot, AIlli

q=tokenizer('<|start_header_id|>user<|end_header_id|> December 29 2002 implosion')['input_ids'] #December 29 2002 implosion 

logits,logp=get_prob(q)

seq=[33246, 220, 1682, 220, 1049, 17, 3242, 41555]

seqs=gen(q,n=5,L=100)

tokenizer.decode(seqs[0])
tokenizer.decode(seqs[1])
tokenizer.decode(seqs[2])
tokenizer.decode(seqs[3])
tokenizer.decode(seqs[4])

model.model.embed_tokens.weight.data[:,:]=F.dropout(model.model.embed_tokens.weight.data,p=0.1,training=True)