from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig
from datasets import Dataset as HF_Dataset
import torch.nn.functional as F


class TrojAIMitigationLLM:
    """
    This is the primary to abstract a TrojAI mitigation on a given Huggingface LLM model. 
    By default, any extra kwargs passed to init will be stored as a keyword attribute in the class.
    
    You may overwrite __init__ in your implementation, but please call super.__init__(device, batch_size, bf16)

    The only function required to implement is mitigate_model, which returns a Huggingface model. 
    """
    def __init__(self, **kwargs):
        pass

    def mitigate_model(self, model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        embed_matrix=[(k,v) for k,v in model.named_parameters() if k.lower().find('embed')>=0]
        print([(k,v.shape) for k,v in embed_matrix])
        v.data[:]=F.dropout(v.data,p=0.2,training=True)
        return model
