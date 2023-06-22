import os
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TextDataset,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from accelerate import Accelerator
from transformers import AdamW
from transformers import EarlyStoppingCallback

accelerator = Accelerator()  # Initialize accelerator

# Define the model and tokenizer
model_name = "/content/drive/MyDrive/Colab Notebooks/models/Main-VergilGPT2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the dataset
dataset_name = "allenai/soda"  # Specify the name of the HuggingFace dataset you want to use
dataset = load_dataset(dataset_name)

# Preprocess the dataset
def preprocess_function(examples):
    # Join dialogue sentences into a single string
    dialogues = [' '.join(dialogue) for dialogue in examples["dialogue"]]
    return tokenizer(dialogues, padding='max_length', truncation=True, max_length=512)

# Preprocess the datasets
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and evaluation sets
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["validation"]  # Assuming the dataset has a validation split

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass the evaluation dataset here
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Function to compute metrics
)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Function to compute metrics
)

# Prepare everything with the accelerator
model, optimizer, train_dataset, trainer = accelerator.prepare(model, optimizer, encoded_dataset, trainer)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
