from accelerate import Accelerator
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Initialize the accelerator
accelerator = Accelerator()

# Define the model and tokenizer
model_name = "/content/drive/MyDrive/Colab Notebooks/models/Vergil-GPT2-main-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("xsum")

# Split the dataset into train, validation, and test sets
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Preprocess function
def preprocess_function(examples):
    inputs = tokenizer(
        examples["document"],
        examples["summary"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    inputs["input_ids"] = inputs["input_ids"].squeeze()
    inputs["attention_mask"] = inputs["attention_mask"].squeeze()
    return inputs

# Apply the preprocess function to the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
validation_dataset = validation_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the output directory path
output_dir = "/content/drive/MyDrive/Colab Notebooks/trained-model"

# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=2,
    fp16=True,  # Enable mixed-precision training
    gradient_accumulation_steps=4,  # Number of update steps to accumulate before performing a backward/update pass.
)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(output_dir)
