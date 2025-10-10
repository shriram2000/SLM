import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# Step 1: Load and prepare the dataset from JSONL
def load_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:  # Added encoding='utf-8'
        for line in f:
            entry = json.loads(line)
            prompt = entry['prompt']
            completion = entry['completion']
            data.append({'prompt': prompt, 'completion': completion})
    return Dataset.from_list(data)
dataset = load_dataset('slm_enduser_dataset_250.jsonl')

# Split into train/test (80/20)
dataset = dataset.train_test_split(test_size=0.2)

# # Step 2: Tokenize the data
model_name = 'distilgpt2'  # Or 'gpt2' for base GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

def tokenize_function(examples):
    # Combine prompt and completion for causal LM (input is prompt + completion)
    full_texts = [p + c for p, c in zip(examples['prompt'], examples['completion'])]
    tokenized = tokenizer(full_texts, truncation=True, max_length=512, padding='max_length')
    tokenized['labels'] = tokenized['input_ids'].copy()  # For causal LM
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)

# Step 3: Set up data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Step 4: Load model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_modeltest',
    num_train_epochs=3,  # Adjust based on dataset size
    per_device_train_batch_size=4,  # Adjust for your GPU
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
)

# Step 7: Fine-tune
trainer.train()

# Step 8: Save the model
trainer.save_model('./fine_tuned_modeltest')
tokenizer.save_pretrained('./fine_tuned_model')
