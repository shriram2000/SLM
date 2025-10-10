# Fine-tuning DistilGPT2 on Custom JSONL Dataset

This script fine-tunes a **DistilGPT2** model using a custom dataset in JSONL format containing user prompts and their expected completions.  
It is designed to train a small language model (SLM) for handling retail end-user queries or similar text generation tasks.

---

## 📂 Input Dataset Format

The dataset must be in **JSONL** format (one JSON per line) with two fields: `prompt` and `completion`.

Example:
```json
{"prompt": "Show me the purchase details for Example Customer (Customer ID 12345) on 2023-12-31 10:00:00.", "completion": "The customer purchased 2 shirts worth $40 and 1 pair of shoes worth $60. Total: $100."}

1️⃣ Load Dataset
----------------
- Opens the `.jsonl` file and reads line by line.
- Extracts `prompt` and `completion` fields.
- Converts them into a Hugging Face `Dataset` object.

2️⃣ Split Dataset
-----------------
- Splits the dataset into:
  • 80% training data
  • 20% testing data

3️⃣ Tokenize Text
-----------------
- Concatenates each `prompt` + `completion`.
- Uses DistilGPT2 tokenizer with:
  • max_length=512
  • truncation=True
  • padding='max_length'
- Sets the pad token to the EOS token.

4️⃣ Data Collator
-----------------
- Uses DataCollatorForLanguageModeling from Transformers.
- Ensures correct batching and dynamic padding.
- Disables masked language modeling (mlm=False).

5️⃣ Load Model
--------------
- Loads the DistilGPT2 base model (AutoModelForCausalLM).
- Model architecture supports causal (next token) prediction.

6️⃣ Define Training Arguments
-----------------------------
- Configures key parameters:
  • num_train_epochs=3
  • batch_size=4 (train & eval)
  • eval_strategy='epoch'
  • save_strategy='epoch'
  • fp16=True (mixed precision for GPU)
  • load_best_model_at_end=True
- Outputs logs and checkpoints to:
  • ./logs/
  • ./fine_tuned_model/

7️⃣ Initialize Trainer
----------------------
- Uses Trainer() from Hugging Face with:
  • Model
  • Tokenized training & test sets
  • Training arguments
  • Data collator

8️⃣ Fine-tune the Model
-----------------------
- Runs the training loop via:
  trainer.train()
- Monitors training loss and evaluation metrics.

9️⃣ Save the Model
------------------
- After fine-tuning, saves:
  • Model weights: ./fine_tuned_model/
  • Tokenizer: ./fine_tuned_model/
