import os

import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM, AutoTokenizer
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model


BASE_MODEL = "facebook/opt-125m"
# BASE_MODEL = "facebook/opt-6.7b"
MICRO_BATCH_SIZE = 4
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 1  
LEARNING_RATE = 3e-4 
CUTOFF_LEN = 512
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
DATA_PATH = "alpaca_data.json"
# DATA_PATH = "alpaca_data_small.json"
ENABLE_16BIT = True

## TODO: Download alpaca_data.json here

if torch.cuda.is_available():
    model = OPTForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        device_map="auto",
    )
    model = prepare_model_for_int8_training(model)
else:
    model = OPTForCausalLM.from_pretrained(
        BASE_MODEL,
    )
    ENABLE_16BIT = False

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    model_max_length=CUTOFF_LEN,
    padding_side="right",
    use_fast=False,
)


config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

data = load_dataset("json", data_files=DATA_PATH)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
{data_point["output"]}"""


data = data.shuffle().map(
    lambda data_point: tokenizer(
        generate_prompt(data_point),
        padding="longest",
        max_length=CUTOFF_LEN,
        truncation=True,
    )
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=ENABLE_16BIT,
        logging_steps=1,
        output_dir="alpaca-opt-6.7b",
        save_total_limit=3,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False
trainer.train(resume_from_checkpoint=False)

model.save_pretrained("alpaca-opt-6.7b")
