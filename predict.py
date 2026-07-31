import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from common import load_tokenizer  # Reuse the tokenizer loader from common.py

# 1. Define absolute path to your round_5 folder
BASE_DIR = "D:/federated_llm_gpu"
CHECKPOINT_PATH = os.path.join(BASE_DIR, "saved_global", "round_5")

print(f"Loading tokenizer...")
# Load tokenizer using your common helper or directly from base model name
tokenizer = load_tokenizer()

print("Loading base model...")
# Replace 'distilgpt2' with whichever base model name you defined in common.py
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

print(f"Loading FL adapter weights from {CHECKPOINT_PATH}...")
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model.eval()

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 2. Get input and run prediction
text = input("\nEnter text: ")
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    next_token_logits = outputs.logits[:, -1, :]
    predicted_id = torch.argmax(next_token_logits, dim=-1)

word = tokenizer.decode(predicted_id)

print("\n--- Output ---")
print("Predicted next word:", word)