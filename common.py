import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "distilgpt2"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.to(device)
    return model