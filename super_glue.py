import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset, DatasetDict, concatenate_datasets

# Define the subsets you want to use
subsets = ['axb', 'subset2', 'subset3']  # replace with actual subset names

# Load the 'test' split of each subset and concatenate them into a single dataset
datasets = []
for subset in subsets:
    dataset = load_dataset('super_glue', subset, split='test')  # changed to 'test'
    datasets.append(dataset)

combined_dataset = concatenate_datasets(datasets)

# Load the SuperGLUE dataset
#dataset = load_dataset('super_glue', 'rte')   # not sure why this line was there, commenting it out

# Using VergilGPT2 Model
model_name = "/content/drive/MyDrive/Colab Notebooks/models/Vergil-GPT2-main-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def encode(examples):
    # Tokenize the inputs and labels
    inputs = tokenizer(examples['premise'], truncation=True, padding='max_length', max_length=512)
    labels = tokenizer(examples['hypothesis'], truncation=True, padding='max_length', max_length=512)['input_ids']
    inputs['labels'] = labels
    return inputs

# Apply the encoding function to all examples in the combined dataset
combined_dataset = combined_dataset.map(encode, batched=True)  # changed to combined_dataset

# Set the format of the dataset to PyTorch tensors
combined_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=combined_dataset,      # training dataset, changed to combined_dataset
    # eval_dataset=dataset['validation'],  # evaluation dataset, commented out as there's no validation split
)

# Train the model
trainer.train()
