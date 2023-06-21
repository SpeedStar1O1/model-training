import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict

# Define the model and tokenizer
model_name = "/content/drive/MyDrive/Vergil-GPT2-conv_ai-1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocess the dataset
def preprocess_dataset(dataset):
    def flatten_dialog(dialog):
        return ' '.join([message['text'] for message in dialog])

    dataset = dataset.map(lambda x: {'dialog': flatten_dialog(x['dialog']),
                                     'bot_profile': x['bot_profile'],
                                     'user_profile': x['user_profile']})
    return dataset


# Load and preprocess the dataset
dataset = load_dataset("conv_ai_2")
dataset = preprocess_dataset(dataset)

# Tokenize inputs
def tokenize_function(examples):
    return tokenizer(examples["dialog"], truncation=True, padding="max_length", max_length=128)

# Tokenize all elements in the dataset
dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and validation sets
dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

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

# Create the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(output_dir)
