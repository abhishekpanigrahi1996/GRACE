from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoTokenizer, HfArgumentParser, AutoConfig
from transformers import TrainingArguments
from arguments import ModelArguments, TrainingArguments, DataArguments
from datasets import load_from_disk
import os
from transformers import Trainer as KLTrainer
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import DataCollatorWithPadding
import torch
import re

import pyarrow.parquet as pq



import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


def manual_collate_fn(batch):
    """
    Collate function that dynamically pads up to `max_sequence_length`
    and truncates longer examples if needed.
    """
    input_ids_list = [ex["input_ids"] for ex in batch]
    labels_list    = [ex["labels"]    for ex in batch]

    # Truncate sequences that exceed max_sequence_length
    input_ids_list = [ids[:max_sequence_length] for ids in input_ids_list]
    labels_list    = [lbl[:max_sequence_length] for lbl in labels_list]

    # Compute the effective max length for this batch
    max_len = min(max(len(ids) for ids in input_ids_list), max_sequence_length)

    pad_token_id = tokenizer.pad_token_id

    def pad(seq, pad_token_id):
        return seq + [pad_token_id] * (max_len - len(seq))

    input_ids = [pad(ids, pad_token_id) for ids in input_ids_list]
    labels    = [pad(lbl, -100)        for lbl in labels_list]
    attn_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in input_ids_list]

    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.long),
        "labels":         torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }
    

def model_exists_and_loadable(path):
    if not os.path.isdir(path):
        return False
    # Check for config.json as minimum
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    # Check for safetensors index OR pytorch_model.bin
    has_safetensors_index = os.path.isfile(os.path.join(path, "model.safetensors.index.json"))
    has_pytorch_bin = os.path.isfile(os.path.join(path, "pytorch_model.bin"))
    if not (has_safetensors_index or has_pytorch_bin):
        return False

    try:
        _ = AutoModelForCausalLM.from_pretrained(path)
        return True
    except Exception as e:
        print(f"Model loading failed with error: {e}")
        return False


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()  

    max_sequence_length=data_args.max_sequence_length

    # if you can load model from output_dir, then you can skip the training
    #if AutoModelForCausalLM.from_pretrained(model_args.output_dir):
    if model_exists_and_loadable(training_args.output_dir):
        print ("Model already exists, skipping training")
        exit()

    model_name = model_args.model_name_or_path
    tokenizer_name = model_args.tokenizer_name
    if tokenizer_name == '':
        tokenizer_name = model_name
    print (tokenizer_name, model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=model_args.cache_dir, padding=True, truncation=True)
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
    )

    if '.hf' in data_args.data_path:
        dataset = load_from_disk(data_args.data_path)
        train_dataset = dataset
    elif '.parquet' in data_args.data_path:
        
        # 3. Now create dataset safely
        dataset = Dataset.from_parquet(data_args.data_path)
        train_dataset = dataset

    else:
        raise NotImplementedError
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    
    training_args.eval_strategy = 'no'
    training_args.report_to = 'tensorboard'


    config = {}
    config['model'] = model
    config['processing_class'] = tokenizer
    config['train_dataset'] = train_dataset
    config['args'] = training_args
    config['data_collator'] = manual_collate_fn
    try:
        trainer = KLTrainer(**config)
    except:
        config.pop('processing_class')
        config['tokenizer'] = tokenizer
        trainer = KLTrainer(**config)
    checkpoint_path = None
    output_dir=training_args.output_dir
    if os.path.exists(output_dir):
        # Look for checkpoint directories or files in output_dir, e.g., "checkpoint-xxxx"
        checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
        def get_checkpoint_index(checkpoint_path):
            # Extract number after 'checkpoint-' prefix
            match = re.search(r'checkpoint-(\d+)', checkpoint_path)
            return int(match.group(1)) if match else -1

        if checkpoints:
            # Sort checkpoints by their numeric index
            checkpoint_path = sorted(checkpoints, key=get_checkpoint_index)[-1]

    if checkpoint_path:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        train_result = trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        train_result = trainer.train()
    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.save_model(training_args.output_dir)  # Saves model & weights

    # Explicitly save the tokenizer
    tokenizer.save_pretrained(training_args.output_dir)