import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Define the model and tokenizer
model_name = "/content/drive/MyDrive/Vergil-GPT2-model-soda datasets" # Be sure to change to your model you want to train
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_dataset(examples):
    # Tokenize examples
    return tokenizer(examples['context'], truncation=True, padding='max_length', max_length=512)

# Load and preprocess the dataset
dataset = load_dataset("allenai/prosocial-dialog")
dataset = dataset.map(preprocess_dataset)

# Extract input_ids and attention_mask from preprocessed dataset and convert to PyTorch tensors
for split in dataset.keys():
    dataset[split].set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Split the dataset into training and validation sets
train_size = 0.9
train_dataset, val_dataset = torch.utils.data.random_split(dataset['train'], [int(train_size * len(dataset['train'])), len(dataset['train']) - int(train_size * len(dataset['train']))])

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
    gradient_accumulation_steps=4,  # Number of updates steps to accumulate before performing a backward/update pass.
)

# Create the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train("/content/drive/MyDrive/Colab Notebooks/trained-model/checkpoint-20000")

# Save the fine-tuned model
trainer.save_model(output_dir)
