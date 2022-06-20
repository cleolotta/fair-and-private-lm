from dataclasses import dataclass, field
from typing import Optional
from transformers import CONFIG_MAPPING, MODEL_WITH_LM_HEAD_MAPPING

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."}
    )
    model_type: Optional[str] = field(
        default="bert", metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
        #angepasst
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    language_train_data_file: Optional[str] = field(
        default="C:/Users/cmatz/master-thesis/fair-and-private-lm/datasets/wikipedia_bookcorpus/augmented-data.txt", metadata={"help": "The input training data file (a text file)."}   
        #angepasst
    )
    language_train_data_file_cda_sep: Optional[str] = field(
        default=None, metadata={"help": "If cda_type=cda_sep: The input training data file (a text file) for second LA training."}
    )
    language_eval_data_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    task_data_files: Optional[str] = field(
        default=None, metadata={"help": "The data files for task training."}
    )
    task_data_files_cda_sep: Optional[str] = field(
        default=None, metadata={"help": "If cda_type=cda_sep: The data files for second TA training."}
    )
    bias_eval_data_file: Optional[str] = field(
        default=None, metadata={"help": "A bias evaluation data file to calculate the bias of a model that is trained on STS-B or MNLI (for raw models or mdoels with a LA: this parameter can be ignored and just the bias_eval_task has to be used). None if no bias calculation desired."}
    )
    bias_eval_task: Optional[str] = field(
        default='bec_pro_english', metadata={"help": "Task on which bias evaluation is done: disco, bec-pro_german, bec_pro_english (not necessary if finetuning/task adapter training on STS-B or MNLI is done: in this case finetuning_task/task_adapter_task is used as bias_eval_task)"}
        #angepasst
    )
    mlm: bool = field(
        default=True, metadata={"help": "Train with masked-language modeling loss instead of language modeling. (Needed for BERT and RoBERTa)"}
        #angepasst
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    block_size: int = field(
        default=128, metadata={"help": "Text datasets (language_train_data_file and language_eval_data_file) are split to blocks of a number of tokens."}
    )
    language_overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class LanguageAdapterArguments:
    """
    Arguments related to language adapter training.
    """
    language_adapter: str = field(
        default=None, metadata={"help": "The name of the language adapter."}
    )
    train_language_adapter: bool = field(
        default=False, metadata={"help": "Train a text language adapter for the given language."}
    )
    la_cda_type: Optional[str] = field(
        default="cda", metadata={"help": "cda or cda-sep for language adapter training: cda for 1-sided or 2-sided and cda-sep for 2-sided-separated."}
    )
    language_adapter_epochs: int = field(
        default=2, metadata={"help": "Total number of training epochs to perform."}
    )
    load_language_adapter: Optional[str] = field(
        default=None, metadata={"help": "Path to pre-trained language adapter. None to create new language adapter."}
    )
    language_adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "The adapter configuration."}
    )
    eval_language_adapter: bool = field(
        default=False, metadata={"help": "True for evaluation of language adapter."}
    )
    language_adapter_batch_size: int = field(
        default=16, metadata={"help": "The batch size per GPU/TPU core/CPU for the language adapter training and evaluation."}
    )
    language_adapter_learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate for Adam for language adapter training."}
    )


@dataclass
class TaskAdapterArguments:
    """
    Arguments related to task adapter training.
    """
    task_adapter: str = field(
        default=None, metadata={"help": "The name of the task adapter."}
    )
    train_task_adapter: bool = field(
        default=False, metadata={"help": "Train a task adapter on the given task."}
    )
    ta_cda_type: Optional[str] = field(
        default="cda", metadata={"help": "cda or cda-sep for task adapter training: cda for 1-sided or 2-sided and cda-sep for 2-sided-separated."}
    )
    task_adapter_task: str = field(
        default=None, metadata={"help": "The task to train the task adapter."}
    )
    task_adapter_epochs: int = field(
        default=2, metadata={"help": "Total number of training epochs to perform."}
    )
    load_task_adapter: Optional[str] = field(
        default=None, metadata={"help": "Path to pre-trained task adapter. None to create new task adapter."}
    )
    task_adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "The adapter configuration."}
    )
    eval_task_adapter: bool = field(
        default=False, metadata={"help": "True for evaluation of task adapter."}
    )
    task_adapter_batch_size: int = field(
        default=32, metadata={"help": "The batch size per GPU/TPU core/CPU for the task adapter training and evaluation."}
    )
    task_adapter_learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate for Adam for task adapter training."}
    )

@dataclass
class FinetuningArguments:
    """
    Arguments related to finetuning.
    """
    do_finetuning: bool = field(
        default=True, metadata={"help": "Do finetuning."}
    )
    finetuning_task: str = field(
        default=None, metadata={"help": "The task to train model on."}
    )
    finetuning_epochs: int = field(
        default=2, metadata={"help": "Total number of training epochs to perform."}
    )
    eval_finetuning: bool = field(
        default=False, metadata={"help": "True for evaluation of finetuned model."}
    )
    finetuning_batch_size: int = field(
        default=32, metadata={"help": "The batch size per GPU/TPU core/CPU for the finetuning and evaluation."}
    )
    finetuning_learning_rate: float = field(
        default=3e-5, metadata={"help": "The initial learning rate for Adam for the finetuning."}
    )

