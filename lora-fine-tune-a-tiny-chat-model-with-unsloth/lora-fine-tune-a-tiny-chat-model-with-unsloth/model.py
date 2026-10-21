"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel
def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.

    Returns:
        (model, tokenizer)
    """
    # TODO: call FastLanguageModel.from_pretrained with 4-bit loading and return (model, tokenizer)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return sum(param.numel() for param in model.parameters())

# Step 3 - is_model_4bit_quantized
import bitsandbytes

def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    for module in model.modules():
        if isinstance(module, bitsandbytes.nn.Linear4bit):
            return True
            
    return False

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    # TODO: if the tokenizer is missing a pad token, reuse its eos token
    if(tokenizer.pad_token is None):
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    # TODO: return the list of attention projection module names LoRA should adapt
    list_of_projections = ['q_proj','k_proj','v_proj','o_proj']

    return list_of_projections

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    
    if(target_modules is None):
        target_modules = get_lora_target_modules()

    model = FastLanguageModel.get_peft_model(
    model,
    r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = target_modules,
    lora_alpha = lora_alpha
    
    )

    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    # TODO: sum p.numel() over model.parameters() where requires_grad is True
    return sum(p.numel() if p.requires_grad else 0 for p in model.parameters())

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    # TODO: return the fraction of parameters that are trainable.
    return trainable_count/total_count

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    # TODO: return a tiny hand-written list of instruction/response example dicts.
    data = [{
        'instruction':'Who is Batman?',
        'response':'Batman is a superhero'
    },
    {
        'instruction':'Tell me about a car',
        'response':'A car is a vehicle with 4 wheels'
    },
    {
        'instruction':'what is 5 + 6',
        'response':'11'
    }]

    return data

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    # TODO: combine example['instruction'] and example['response'] into one string

    return f'### Instruction:\n{example["instruction"]}\n\n### Response:\n{example["response"]}'

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    # TODO: apply format_instruction_example to every example and return the list
    for i in range(len(examples)):
        examples[i] = format_instruction_example(examples[i])
    
    return examples

# Step 12 - build_text_dataset
from datasets import Dataset
def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    # TODO: return a datasets.Dataset with one 'text' column holding the given strings
    data = Dataset.from_dict({'text':texts})
    return data

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    # TODO: call the tokenizer on text and return its input_ids as a plain list
    return tokenizer(text)['input_ids']

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    # TODO: return the length of the input_ids sequence
    return len(input_ids)

# Step 15 - build_training_arguments
from transformers import TrainingArguments
import torch

def build_training_arguments(
    output_dir="./sft_out",
    max_steps=5,
    learning_rate=2e-4
):
    bf16 = torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        learning_rate=learning_rate,
        gradient_accumulation_steps=1,
        bf16=bf16,
        fp16=fp16,
        logging_steps=1,
        optim="adamw_8bit",
        save_strategy="no",
        report_to="none",
    )

    return training_args

# Step 16 - build_sft_trainer
# from trl import SFTTrainer
import trl
def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    
    trainer = trl.SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        tokenizer=tokenizer,
        save_strategy="no",
        args=training_args,
        max_seq_length=max_seq_length,
    )
    return trainer

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    train_result = trainer.train()
    return float(train_result.training_loss)

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    # TODO: call the Unsloth helper that prepares the model for fast generation
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    return model

# Step 19 - build_chat_prompt
def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# Step 20 - generate_reply
def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    # Tokenize and move to the model's active device, returning PyTorch tensors
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Run generation with greedy decoding (do_sample=False)
    outputs = model.generate(
        **inputs, 
        do_sample=False, 
        max_new_tokens=max_new_tokens
    )
    
    # Slice out the original prompt tokens so we only decode the newly generated response
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

