import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                          BitsAndBytesConfig, DataCollatorForLanguageModeling, 
                          TrainingArguments, Trainer
                          )
from datasets import load_dataset
from peft import prepare_model_for_kbit_training

# Define the model and tokenizer
model_name = "gpt2-medium"

# Define the BitsAndBytesConfig for QLoRA
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model using QLoRA
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=nf4_config, device_map={"":0})

# Prepare model for kbit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Define the dataset
dataset_name = "allenai/soda"
dataset = load_dataset(dataset_name)

# Preprocess the dataset
def preprocess_function(examples):
    dialogues = [' '.join(dialogue) for dialogue in examples["dialogue"]]
    return tokenizer(dialogues, padding='max_length', truncation=True, max_length=512)

# Preprocess the training dataset
encoded_dataset = dataset["train"].map(preprocess_function, batched=True)

# needed for gpt2 tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Define the training arguments
training_args = TrainingArguments(
    output_dir="trained-models",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,  # Increased batch size
    gradient_accumulation_steps=2,  # Using gradient accumulation
    learning_rate=5e-5,  # Increased learning rate
    fp16=True,  # Enable mixed precision training
    dataloader_num_workers=4,  # Number of subprocesses for data loading
    optim="paged_adamw_8bit"  # QLoRA uses paged_adamw_8bit optimizer
)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Set use_cache to False (required for kbit training)
model.config.use_cache = False

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
