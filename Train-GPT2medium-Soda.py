import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TextDataset, DataCollatorForLanguageModeling, 
    TrainingArguments, Trainer
)
from accelerate import Accelerator

# Initialize the accelerator
accelerator = Accelerator()

# Define the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the dataset
dataset_name = "allenai/soda"
dataset = load_dataset(dataset_name)

# Preprocess the dataset
def preprocess_function(examples):
    # Join dialogue sentences into a single string
    dialogues = [' '.join(dialogue) for dialogue in examples["dialogue"]]
    return tokenizer(dialogues, padding='max_length', truncation=True, max_length=512)

# Preprocess the training dataset
encoded_dataset = dataset["train"].map(preprocess_function, batched=True)

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

# Prepare the training
model, train_dataset = accelerator.prepare(model, encoded_dataset)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
