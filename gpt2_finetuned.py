import os
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2TokenizerFast,
    GPT2LMHeadModel,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
import re

# settings
CSV_PATH = "species_with_description_fixed.csv"
MODEL_NAME = "gpt2"
OUTPUT_DIR = "gpt2-finetuned-binomial"
MAX_LENGTH = 64
BATCH_SIZE = 8
EPOCHS = 10
LR = 5e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load CSV
df = pd.read_csv(CSV_PATH)

def extract_genus_epithet(row):
    name = row.get("canonicalName") if pd.notna(row.get("canonicalName")) else row.get("scientificName", "")
    parts = str(name).split()
    if len(parts) >= 2:
        genus, epithet = parts[0], parts[1]
    else:
        genus = ""
        epithet = row.get("epithet", "")
    return genus.strip(), str(epithet).strip()
rows = []
for _, r in df.iterrows():
    genus, epithet = extract_genus_epithet(r)
    description = r.get("description", "")
    family = r.get("family", "")
    if not genus or not epithet or not description:
        continue
    prompt = f"Description: {description.strip()}\nFamily: {family.strip()}\nName:"
    target = f" {genus} {epithet}"
    rows.append({"prompt": prompt, "target": target, "genus": genus, "epithet": epithet})

# Tokenizer & Model
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.to(DEVICE)

# Dataset
class BinomialDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=MAX_LENGTH):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["prompt"]
        target = ex["target"]
        full = prompt + target
        enc = self.tokenizer(
            full,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        enc_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt_len = enc_prompt["input_ids"].size(1)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "genus": ex["genus"], "epithet": ex["epithet"]}

train_exs, val_exs = train_test_split(rows, test_size=0.05, random_state=SEED)
train_dataset = BinomialDataset(train_exs, tokenizer)
val_dataset = BinomialDataset(val_exs, tokenizer)

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_eval=True,
    eval_steps=500,
    save_steps=500,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_steps=100,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Latinized Epithet Constraint
LATIN_EPITHET_REGEX = re.compile(r"^[a-z]+(us|a|um|is|ensis|ii)?$")

def latin_epithet_allowed_tokens_fn(batch_id, input_ids_so_far):
    decoded = tokenizer.decode(input_ids_so_far, skip_special_tokens=True)
    if "Name:" in decoded:
        name_part = decoded.split("Name:")[-1].strip()
    else:
        name_part = ""
    words = name_part.split()
    vocab = list(range(len(tokenizer)))
    if len(words) >= 2:
        return [tokenizer.eos_token_id]
    if len(words) == 0:
        return vocab
    allowed_ids = []
    for token_id in vocab:
        candidate = tokenizer.decode([token_id]).strip().lower()
        if LATIN_EPITHET_REGEX.match(candidate):
            allowed_ids.append(token_id)
    return allowed_ids if allowed_ids else vocab  

# Example generation
model.eval()
example_prompt = "Description: a small white bear\nFamily: Ursidae\nName: "
input_ids = tokenizer(example_prompt, return_tensors="pt").input_ids.to(DEVICE)

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 40,
        num_beams=5,
        do_sample=False,
        prefix_allowed_tokens_fn=latin_epithet_allowed_tokens_fn,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping=True,
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Prompt:")
print(example_prompt)
print("Generated scientific name:")
print(generated_text.split("Name:")[-1].strip())
