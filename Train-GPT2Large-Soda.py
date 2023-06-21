from accelerate import Accelerator
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TextDataset, 
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer, AdamW, BitsAndBytesConfig)

# Initialize the accelerator
accelerator = Accelerator()

# Define the model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model using QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path=model_name,
    load_in_4bit=True,
    device_map='auto',
    max_memory=48,  # Change this depending on your system
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)

# Define the dataset
dataset_name = "allenai/soda"
dataset = load_dataset(dataset_name)

# Preprocess the dataset
def preprocess_function(examples):
    dialogues = [' '.join(dialogue) for dialogue in examples["dialogue"]]
    return tokenizer(dialogues, padding='max_length', truncation=True, max_length=512)

# Preprocess the training dataset
encoded_dataset = dataset["train"].map(preprocess_function, batched=True)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="trained-models",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Increased batch size
    gradient_accumulation_steps=2,  # Using gradient accumulation
    save_steps=10000,
    save_total_limit=2,
    learning_rate=5e-5,  # Increased learning rate
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,  # Number of subprocesses for data loading
    # QLoRA uses paged_adamw_32bit optimizer
    optimization_strategy="epoch",
    optimizer_type="paged_adamw_32bit",
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
