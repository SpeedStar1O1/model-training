from accelerate import Accelerator
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    AdamW,
    get_linear_schedule_with_warmup
)

# Initialize the accelerator with fp16 for faster training
accelerator = Accelerator(fp16=True)

# Define the model and tokenizer
model_name = "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# Enable gradient checkpointing to save memory
model.config.gradient_checkpointing = True

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
    output_dir="/content/drive/MyDrive/Colab Notebooks/trained-model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Increase this if your GPU has enough memory
    gradient_accumulation_steps=16,  # Increase gradient accumulation to simulate larger batch size
    save_steps=10000,
    save_total_limit=2,
    learning_rate=5e-5,  # Experiment with different learning rates
)

# Use a learning rate scheduler
optimizer = AdamW(model.parameters(), lr=1e-5)
total_steps = training_args.num_train_epochs * (len(encoded_dataset) // training_args.per_device_train_batch_size)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Prepare the training
model, train_dataset = accelerator.prepare(model, encoded_dataset)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizer=optimizer,  # Use custom optimizer with scheduler
    scheduler=scheduler  # Add learning rate scheduler
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
