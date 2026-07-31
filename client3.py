import flwr as fl
import torch
from common import load_model, load_tokenizer, device
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "client3.txt")
tokenizer = load_tokenizer()
model = load_model()

class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        texts = [line.strip() for line in lines if line.strip()]

        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=32
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.encodings["input_ids"][idx]),
        }

def get_dataset():
    return MyDataset(DATA_FILE, tokenizer)
def train():
    dataset = get_dataset()

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
    output_dir="./tmp3",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    logging_steps=5,
    save_strategy="no",
    report_to="none"
),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
    )

    trainer.train()

class Client(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [v.cpu().detach().numpy() for v in model.parameters() if v.requires_grad]

    def set_parameters(self, parameters):
        trainable = [p for p in model.parameters() if p.requires_grad]
        for p, new in zip(trainable, parameters):
            p.data = torch.tensor(new).to(device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train()
        return self.get_parameters(config), 1, {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {"accuracy": 0.0}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=Client()
)