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
model_name = "SpeedStar101/VergilGPT2"
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

# Preprocess the training dataset
encoded_dataset = dataset["train"].map(preprocess_function, batched=True)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {'loss': torch.nn.functional.cross_entropy(logits, labels)}

output_dir = "/content/drive/MyDrive/Colab Notebooks/models/Vergil-GPT2-main-model"
# Define the training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10000,
    save_total_limit=2,
    fp16=True,  # Enable mixed-precision training
    gradient_accumulation_steps=4,  # Number of updates steps to accumulate before performing a backward/update pass.
    load_best_model_at_end=True,  # Load the best model at the end
    evaluation_strategy = 'steps',  # Add this line
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


# Check if a checkpoint exists
checkpoint_dir = "/content/drive/MyDrive/Colab Notebooks/trained-model/checkpoints/Soda checkpoints/checkpoint-220000"  # Modify with your checkpoint path
checkpoint = None
if os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir):
    checkpoint = checkpoint_dir

# Prepare everything with the accelerator
model, optimizer, train_dataset, trainer = accelerator.prepare(model, optimizer, encoded_dataset, trainer)

# Fine-tune the model
trainer.train(resume_from_checkpoint=checkpoint)

# Save the fine-tuned model
trainer.save_model(training_args.output_dir)
