from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import TrainingArguments as TA

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='',
        metadata={"help": "Path to the teacher model"}
    )
    tokenizer_name: str = field(
        default='',
        metadata={"help": "Path to the teacher model"}
    )
    
    cache_dir: Optional[str] = field(
        default='hf_models',
        metadata={"help": "cache dir"}
    )
    pad_token_id: Optional[int] = field(
        default=-1,
        metadata={"help": "Size of vocabulary"}
    )
    


@dataclass
class TrainingArguments(TA):
    """
    Arguments pertaining to training arguments for distillation
    """

    seed: Optional[int] = field(
        default=42,
        metadata={"help": "Seed to use for training (including data shuffling)"}
    )

    project_name: Optional[str] = field(
        default='GRACE',
        metadata={"help": "Name of the W&B project"}
    )

    report_to: Optional[str] = field(
        default='wandb',
        metadata={"help": "Whether to log to W&B"}
    )

    disable_tqdm: Optional[bool] = field(
        default=True,
        metadata={"help": "Disable TQDM for cleaner logging"}
    )

    shuffle_train: Optional[bool] = field(
        default=False,
        metadata={"help": "Shuffle training data"}
    )

  

@dataclass
class DataArguments:
    #dataset to train on
    data_path: Optional[str] = field(
        default='',
        metadata={"help": " Path to the dataset"}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_sequence_length: Optional[int] = field(
        default=4096,
        metadata={"help": "Context length"}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class Unionable:
    def __or__(self, other):
        return self.__class__(**asdict(self) | asdict(other))