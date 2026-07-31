# Federated-LLM-LoRA-Simulation
# Federated LLM Fine-Tuning Simulation using Flower, PyTorch & PEFT (LoRA)

A lightweight, end-to-end simulation of Federated Learning (FL) for Causal Language Models using **Flower (`flwr`)**, **PyTorch**, **Hugging Face Transformers**, and **PEFT (LoRA)**. 

This repository demonstrates how multiple local clients can collaboratively fine-tune a language model (e.g., `distilgpt2`) using LoRA adapters while keeping local training data completely private on each client device.

---

## 🌟 Key Features

- **🌐 Federated Aggregation:** Custom server strategy (`SaveStrategy`) extending `FedAvg` to aggregate model weights across clients and export PyTorch/PEFT checkpoints after each round.
- **⚡ Parameter-Efficient Fine-Tuning (PEFT):** Uses LoRA adapters to drastically cut down bandwidth and hardware requirements during client-server transfers.
- **🔒 Data Privacy & Independence:** Each client trains locally on its own custom text dataset without transmitting raw data to the central server.
- **🔮 Next-Word Inference Script:** Includes a dedicated `predict.py` script to test next-word generation on the aggregated global checkpoint.

---

## 📁 Repository Structure

```text
.
├── server.py               # Central Flower server & custom aggregation strategy
├── client1.py              # Local training script for Client 1
├── client2.py              # Local training script for Client 2
├── client3.py              # Local training script for Client 3
├── common.py               # Shared utility to load base model & tokenizer
├── predict.py              # Inference script for testing saved global models
├── data/                   # Directory reserved for client text files (Git ignored)
│   ├── client1.txt         # (User-provided text data for Client 1)
│   ├── client2.txt         # (User-provided text data for Client 2)
│   └── client3.txt         # (User-provided text data for Client 3)
└── README.md               # Project documentation
