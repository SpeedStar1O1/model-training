import os
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TextDataset,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from accelerate import Accelerator
from transformers import AdamW
from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split

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

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/Colab Notebooks/trained-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=2,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {'loss': torch.nn.functional.cross_entropy(logits, labels)}

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Split the dataset into training and validation sets
train_dataset = encoded_dataset["train"]
eval_dataset = encoded_dataset["validation"]  # Assuming the dataset has a validation split

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Pass the evaluation dataset here
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # Function to compute metrics
)

# Prepare everything with the accelerator
model, optimizer, train_dataset, trainer = accelerator.prepare(model, optimizer, train_dataset, trainer)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
