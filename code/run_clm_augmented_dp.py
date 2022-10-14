#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# for dp_cda and cda
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import opacus
from opacus.utils.module_modification import convert_batchnorm_modules
import argparse
from enum import unique
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import copy 
from sys import path
import sys
from utils import Logger
import dp_transformers
import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
#    CONFIG_MAPPING,
#    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)


#from torch import AdamW
from transformers.utils.versions import require_version
from transformers.testing_utils import CaptureLogger
import datasets
from datasets import load_dataset
from random import shuffle
import numpy as np
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import csv
from scipy.stats import skewnorm
from scipy.stats import kstest
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora
from data_prep import cda_words
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

#MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
#MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--do_ref_model",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--add_dp",
        action="store_true",
        help="If passed, will add differential privacy to model",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
   #     choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument("--train_head_only",action="store_true", help = "If true, freeze all the layers except the head of the model.")
    parser.add_argument("--train_layer_n_only",default=None, type=int,  help = "If true, freeze all the layers except the n'th layer of the model.")
    parser.add_argument("--lora_dim", default=0, type=int,  help= "LoRA dimension; 0 means LoRA is disabled")
    parser.add_argument("--lora_dropout", default=0.0, type=float,  help= "Dropout probability for LoRA layers")
    parser.add_argument('--lora_alpha', default=32, type=int,  help="LoRA attention alpha")
    parser.add_argument('--per_sample_max_grad_norm', default=0.0, type=float)
    parser.add_argument('--noise_multiplier', default=None, type=float)
    parser.add_argument('--objective', default=None, type=str)
    parser.add_argument('--counterfactual_augmentation', default='gender')
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    return args

            
def main():

    args = parse_args()
    random.seed(args.seed)


    folder_name = f"objective_{args.objective}_add_dp_{args.add_dp}_lora_{args.lora_dim}_noise_multiplier_{args.noise_multiplier}_ref_{args.do_ref_model}_maxlen_{args.block_size}_model_{args.model_name_or_path}_lr_{args.learning_rate}_epoch_{args.num_train_epochs}_trba_{args.per_device_train_batch_size}_acc_{args.gradient_accumulation_steps}_evba{args.per_device_eval_batch_size}"
    
    directory = "{}/{}".format(args.output_dir,folder_name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    log_file = os.path.join(directory, "stdout")


    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    
    if accelerator.is_local_main_process:
        print("Logging to {}".format(log_file))
        
    sys.stdout = Logger(log_file)

        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    directory = os.path.join(directory,"model")

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(directory, exist_ok=True)
    accelerator.wait_for_everyone()
    
    if accelerator.is_local_main_process:
       print(str(args))
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
    extension = args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
        dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)#, cache_dir= "/storage/ukp/work/matzken/fplm/ft_gpt2/cache")
    #print(raw_datasets['train']['text'][0])
    txt = Path(args.train_file).resolve()
    original_train_file = sum(1 for row in open(txt, "r", encoding= "utf-8"))
    txt1 = Path(args.validation_file).resolve()
    original_eval_file = sum(1 for row in open(txt1, "r", encoding= "utf-8"))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)#, cache_dir="/storage/ukp/work/matzken/fplm/ft_gpt2/cache")
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)#, cache_dir="/storage/ukp/work/matzken/fplm/ft_gpt2/cache")
#    else:
#        config = CONFIG_MAPPING[args.model_type]()
#        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)#, not args.use_slow_tokenizer, skip_special_tokens = True)

    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, skip_special_tokens = True)#, add_special_tokens = False)

    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    # Set padding token.
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = config.eos_token_id
    
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    
    model_ref = copy.deepcopy(model)


    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )
    
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output
    
    #raw_text = raw_datasets['train']['text']
    #tokenized_test = tokenize_function(raw_datasets['train'])
    
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )   

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    #group_texts(tokenized_datasets['train']['text'])
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    
    def create_mia_dataset(examples, bias_word_list):
        outputs = []
        #original =  []
        for input_ids in examples["input_ids"]:
            # For simplicity, decode each example. It is easier to apply augmentation
            # on text as opposed to token IDs.
            sentence = tokenizer.decode(input_ids)
            words = sentence.split()  # Tokenize based on whitespace.
            augmented_sentence = words[:]

            augmented = False
            for position, word in enumerate(words):
                if word in bias_word_list:
                    augmented = True
                    break

            if augmented:
                outputs.append(sentence)
                outputs.append(sentence) # append sentence twice for membership inference attack later
            else:
                outputs.append(sentence)

        # There are potentially no counterfactual examples.
        if not outputs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        

        result = tokenizer(
            outputs,
            return_special_tokens_mask=False,
            add_special_tokens=False, 
            truncation=True,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result 

    def gender_counterfactual_augmentation(examples, bias_attribute_words):
        """Applies gender counterfactual data augmentation to a batch of examples.
            Notes:
            * We apply CDA after the examples have potentially been grouped.
            * This implementation can be made more efficient by operating on
              token IDs as opposed to text. We currently decode each example
              as it is simpler.
        """
        outputs = []
        #original =  []
        for input_ids in examples["input_ids"]:
            # For simplicity, decode each example. It is easier to apply augmentation
            # on text as opposed to token IDs.
            sentence = tokenizer.decode(input_ids)
            words = sentence.split()  # Tokenize based on whitespace.
            augmented_sentence = words[:]

            augmented = False
            for position, word in enumerate(words):
                for word_pair in bias_attribute_words:
                    if word == word_pair[0]:
                        augmented = True
                        augmented_sentence[position] = word_pair[1]

            if augmented:
                augmented_sentence = " ".join(augmented_sentence)
                outputs.append(augmented_sentence)
                outputs.append(sentence)
                #original.append(sentence) # append sentence twice for membership inference attack later
                #original.append(sentence)
            else:
                outputs.append(sentence)
                #original.append(sentence)

        # There are potentially no counterfactual examples.
        if not outputs:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        

        augmented = tokenizer(
            outputs,
            return_special_tokens_mask=False,
            add_special_tokens=False,  # Special tokens are already added.
            truncation=True,
            padding="max_length",
        )
        augmented["labels"] = augmented["input_ids"].copy()
        return augmented
    
    
    
    column_names = lm_datasets['train'].column_names
    
    if args.counterfactual_augmentation is not None:
        
        logger.info(f"Applying {args.counterfactual_augmentation} CDA.")

        # Load the bias attribute words.
        print("Get gender word pairs...")
        word_pairs = cda_words.get_gender_word_pairs()
        print("...done\n")
        bias_word_list = cda_words.get_gender_word_list()
        counterfactual_augmentation_func = partial(
            gender_counterfactual_augmentation,
            bias_attribute_words=word_pairs,
        )

        tokenized_datasets = lm_datasets.map(
            counterfactual_augmentation_func,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=column_names,
            desc=f"Applying counterfactual augmentation",
        )
        mia_dataset_creation = partial(
            create_mia_dataset,
            bias_word_list=bias_word_list,
        )
        mia_tokenized_datasets = lm_datasets.map(
            mia_dataset_creation,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=column_names,
            desc=f"Creating a dataset for membership inference attack (loss comparison of augmented and not augmented sentence)",
        )
        
    mia_train_dataset = mia_tokenized_datasets['train']
    mia_eval_dataset = mia_tokenized_datasets['validation']    
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']
        
    # for differential privacy:
    if args.add_dp:
        if args.lora_dim > 0:
            model = convert_gpt2_attention_to_lora(
                model, r=args.lora_dim, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                enable_lora=[True, False, True], merge_weights=False
            )
            mark_only_lora_as_trainable(model)
            if args.lora_dim > 0:
                dp_transformers.register_grad_sampler_gpt2_lora()
            else:
                dp_transformers.register_grad_sampler_gpt2()
        
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    #print("------------------------------------ make sure the layers are frozen ---------------------------------------------------------------")
    #for params in model.parameters():
    #    print(params.requires_grad)
    #print("----------------------------- make sure the layers of ref model are not frozen -----------------------------------------------------")
    #for params in model_ref.parameters():
    #    print(params.requires_grad)

    
    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)
    if args.add_dp:
        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    else:
        data_collator = default_data_collator
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    mia_train_dataloader = DataLoader(
        mia_train_dataset, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    mia_eval_dataloader = DataLoader(
        mia_eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
    )
    if args.train_head_only:
        for params in model.parameters():
            params.requires_grad = False

        for param in model.lm_head.parameters():
            param.requires_grad = True
    
    
    elif args.train_layer_n_only is not None:
        n = args.train_layer_n_only
        k = 0
        for params in model.parameters():
                params.requires_grad = False
        
        for params in model.transformer.h[n].parameters():
                params.requires_grad = True
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, mia_train_dataloader, mia_eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, mia_train_dataloader, mia_eval_dataloader
    )

    model_ref = accelerator.prepare(
        model_ref
    )
         
    # for privacy objective:
    if args.add_dp:
        model = model.train()
        sampling_probability = args.per_device_train_batch_size*accelerator.num_processes*args.gradient_accumulation_steps/len(train_dataset)
        num_steps = int(args.num_train_epochs*(1/sampling_probability+1))
        if args.noise_multiplier is None: 
            noise_multiplier = dp_transformers.dp_utils.find_noise_multiplier(
            sampling_probability=sampling_probability,
            num_steps=num_steps,
            target_delta=1.0/len(train_dataset),
            target_epsilon=args.target_epsilon
        )
        else:
            noise_multiplier = args.noise_multiplier
        # enter PrivacyEngine
        privacy_engine = opacus.PrivacyEngine(module=model,
            batch_size=args.per_device_train_batch_size*args.gradient_accumulation_steps, sample_size=len(train_dataset),
            max_grad_norm=1.0, noise_multiplier=noise_multiplier, target_delta=1.0/len(train_dataset)
        ) # default values from https://github.com/microsoft/dp-transformers
        privacy_engine.attach(optimizer)
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
# Here freezing of the model except the last layer i.e the head is performed
    if accelerator.is_local_main_process:
        print("model_params (million)", count_parameters(model)/1000000)

    

    logger.info("***** Running training *****")
    logger.info(f"  Num original dataset train examples = {original_train_file}")
    logger.info(f"  Num original dataset validation examples = {original_eval_file}")

    logger.info(f"  Num og-mia train examples = {len(mia_train_dataset)}")
    logger.info(f"  Num og-mia eval examples = {len(mia_eval_dataset)}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num eval examples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    with open("dataset_v2.txt",'a+', encoding='utf-8') as f:
        for step, batch in enumerate(train_dataloader):
        #print(batch)
            f.write(tokenizer.decode(batch['input_ids'][0]))
            f.write("\n ---------- \n")
            if step == 100:
                break

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_loss = 1000000
    for epoch in range(args.num_train_epochs):
        model.train()
        if accelerator.is_local_main_process:
            print(f"training epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            print(tokenizer.decode(batch['input_ids'][0]))
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % args.eval_steps == 0:
                        model.eval()
                        losses = []
                        for step, batch in enumerate(eval_dataloader):
                            with torch.no_grad():
                                outputs = model(**batch)

                            loss = outputs.loss
                            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
                            

                        losses = torch.cat(losses)
                        losses = losses[: len(eval_dataset)]
                        try:
                            perplexity = math.exp(torch.mean(losses))
                        except OverflowError:
                            perplexity = float("inf")
                       
                        if torch.mean(losses) < best_loss:
                            best_loss=torch.mean(losses)
                            if accelerator.is_local_main_process:
                                print(f"saving model here at step {completed_steps} and epoch {epoch} with ppl {perplexity}")
                            
                            if args.output_dir is not None:
                                accelerator.wait_for_everyone()
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(directory, save_function=accelerator.save)
                                if accelerator.is_main_process:
                                    tokenizer.save_pretrained(directory)   
                            
                        if accelerator.is_local_main_process:
                            print(f"step {completed_steps} epoch {epoch} perplexity: {perplexity}")
                        #exit()
                        model.train()
            
            if completed_steps >= args.max_train_steps:
                break   
        model.eval()
        losses = []
        if accelerator.is_local_main_process:
            print(f"*************end of epoch {epoch} eval ")
        
        
        if args.do_ref_model:
            model_ref.eval()
            losses_ref = []
            
        for i, (batch1, batch2) in enumerate(zip(eval_dataloader, mia_eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch1)
                

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        
            if args.do_ref_model:
            #evaluate reference model on not-augmented dataset
                with torch.no_grad():
                    outputs_ref =model_ref(**batch2)
                loss_ref = outputs_ref.loss
                losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_eval_batch_size)))
            

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
     
        
        if args.do_ref_model:
            losses_ref = torch.cat(losses_ref)
            losses_ref = losses_ref[: len(mia_eval_dataset)]
            sorted_ratio = sorted([l/l_ref for l,l_ref in zip (losses,losses_ref)])
        
        sorted_loss = sorted(losses)
        
        if args.do_ref_model:
            threshold_ref = sorted_ratio[int(0.1*len(sorted_ratio))]
            threshold = sorted_loss[int(0.1*len(losses))]
            if accelerator.is_local_main_process:
                print("threshold_ref is: " , threshold_ref.detach().item())
                print("threshold is: " , threshold.detach().item())
        else:
            threshold = sorted_loss[int(0.1*len(losses))]
            if accelerator.is_local_main_process:
                print("threshold is: " , threshold.detach().item())
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")
        
        if torch.mean(losses) < best_loss:
            best_loss=torch.mean(losses)
            if accelerator.is_local_main_process:
                print(f"saving model here at step {completed_steps} and epoch {epoch} with ppl {perplexity}")
               
                if args.output_dir is not None:
                    if accelerator.is_main_process:
                        os.makedirs(directory+f"/best", exist_ok=True)
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(directory+f"/best", save_function=accelerator.save)

                    
        if args.output_dir is not None:
            if accelerator.is_main_process:
                os.makedirs(directory+f"/epoch_{epoch}", exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(directory+f"/epoch_{epoch}", save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(directory)   
          
        ################################################    
        #run threshold on training samples
        losses = []
        if args.do_ref_model:
            model_ref.eval()
            losses_ref = []
            
        for i, (batch1, batch2) in enumerate(zip(train_dataloader, mia_train_dataloader)):
            with torch.no_grad():
                outputs = model(**batch1)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.per_device_train_batch_size)))
        
            if args.do_ref_model:
                with torch.no_grad():
                    outputs_ref =model_ref(**batch2)
                loss_ref = outputs_ref.loss
                losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_train_batch_size)))
            
        

        accelerator.wait_for_everyone()
        losses = torch.cat(losses)
        losses = losses[: len(train_dataset)]

        
        if args.do_ref_model:
            losses_ref = torch.cat(losses_ref)
            losses_ref = losses_ref[: len(mia_train_dataset)]
            lr_rat = [l/l_r for l,l_r in zip(losses,losses_ref)]
            
        if args.do_ref_model:
            guess_cor = sum([1 for sample in losses if sample<threshold])
            guess_cor_ref =  sum([1 for sample in lr_rat if sample<threshold_ref])
        else:    
            guess_cor = sum([1 for sample in losses if sample<threshold])


        
        try:
            perplexity_train = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity_train = float("inf")
        #assert(len(losses)==len(lr_rat))
        if accelerator.is_local_main_process:
            if args.do_ref_model:
                print("correct cnt  ref is: " , guess_cor_ref, "all is: ", len(losses), "ratio is: ", guess_cor_ref/len(losses))
            print("correct cnt is: " , guess_cor, "all is: ", len(losses), "ratio is: ", guess_cor/len(losses))
            print(f"epoch {epoch}: perplexity: {perplexity} perplexity_train: {perplexity_train}")
            print("____")
            if args.do_ref_model:
                print(f"{guess_cor_ref/len(losses)}\n{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
                ratio = len(mia_train_dataset)/len(mia_eval_dataset)
                guess_cor_subsampled = sum([1 for sample in losses[::int(ratio)] if sample<threshold])
                guess_cor_ref_subsampled =  sum([1 for sample in lr_rat[::int(ratio)] if sample<threshold_ref])                
                print(f"{guess_cor_ref_subsampled/(len(lr_rat[::int(ratio)]))}\n{guess_cor_subsampled/len(losses[::int(ratio)])}\n{guess_cor_ref_subsampled/(guess_cor_ref_subsampled+int(0.1*len(mia_eval_dataset)))}\n{guess_cor_subsampled/(guess_cor_subsampled+int(0.1*len(mia_eval_dataset)))}")

            else:
                print(f"{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
            print("_____")
        if args.add_dp:
            eps, alpha = optimizer.privacy_engine.get_privacy_spent(1.0/len(train_dataset))
            print("End of epoch {}, we have epsilon {} for alpha {} from privacy engine".format(epoch, eps, alpha))

    model.eval()
    losses = []
    if accelerator.is_local_main_process:
        print(f"*************end of training ")
    
    if args.do_ref_model:
        model_ref.eval()
        losses_ref = []
        
    for i, (batch1, batch2) in enumerate(zip(eval_dataloader, mia_eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch1)
            

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        
        if args.do_ref_model:
            with torch.no_grad():
                outputs_ref =model_ref(**batch2)
            loss_ref = outputs_ref.loss
            losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_eval_batch_size)))
    
    
    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    
    if args.do_ref_model:
        losses_ref = torch.cat(losses_ref)
        losses_ref = losses_ref[: len(mia_eval_dataset)]
        sorted_ratio = sorted([l/l_ref for l,l_ref in zip (losses,losses_ref)])
    
    sorted_loss = sorted(losses)
    
    if args.do_ref_model:
        threshold_ref = sorted_ratio[int(0.1*len(sorted_ratio))]
        threshold = sorted_loss[int(0.1*len(losses))]
        if accelerator.is_local_main_process:
            print("threshold_ref is: " , threshold_ref.detach().item())
            print("threshold is: " , threshold.detach().item())
    else:
        threshold = sorted_loss[int(0.1*len(losses))]
        if accelerator.is_local_main_process:
            print("threshold is: " , threshold.detach().item())
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    
    #run threshold on training samples
    losses = []
    if args.do_ref_model:
        model_ref.eval()
        losses_ref = []
        
    for i, (batch1, batch2) in enumerate(zip(train_dataloader, mia_train_dataloader)):
        with torch.no_grad():
            outputs = model(**batch1)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(args.per_device_train_batch_size)))
        
        if args.do_ref_model:
            with torch.no_grad():
                outputs_ref =model_ref(**batch2)
            loss_ref = outputs_ref.loss
            losses_ref.append(accelerator.gather(loss_ref.repeat(args.per_device_train_batch_size)))
        
    
    accelerator.wait_for_everyone()
    losses = torch.cat(losses)
    losses = losses[: len(train_dataset)]
    
    
    if args.do_ref_model:
        losses_ref = torch.cat(losses_ref)
        losses_ref = losses_ref[: len(mia_train_dataset)]
        lr_rat = [l/l_r for l,l_r in zip(losses,losses_ref)]
        
    if args.do_ref_model:
        guess_cor = sum([1 for sample in losses if sample<threshold])
        guess_cor_ref =  sum([1 for sample in lr_rat if sample<threshold_ref])
    else:    
        guess_cor = sum([1 for sample in losses if sample<threshold])


    
    try:
        perplexity_train = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity_train = float("inf")
    if args.add_dp:
        eps_rdp, alpha = privacy_engine.get_privacy_spent(1.0/len(train_dataset))
        print(f"end of training epsilon privacy engine {eps_rdp}")
    if accelerator.is_local_main_process:
        if args.do_ref_model:
            print("correct cnt  ref is: " , guess_cor_ref, "all is: ", len(losses), "ratio is: ", guess_cor_ref/len(losses))
        print("correct cnt is: " , guess_cor, "all is: ", len(losses), "ratio is: ", guess_cor/len(losses))
        print(f"end of training perplexity: {perplexity} perplexity_train: {perplexity_train}")
        print("____")
        if args.do_ref_model:
            print(f"{guess_cor_ref/len(losses)}\n{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
            ratio = len(train_dataset)/len(eval_dataset)
            guess_cor_subsampled = sum([1 for sample in losses[::int(ratio)] if sample<threshold])
            guess_cor_ref_subsampled =  sum([1 for sample in lr_rat[::int(ratio)] if sample<threshold_ref])                
            print(f"{guess_cor_ref_subsampled/(len(lr_rat[::int(ratio)]))}\n{guess_cor_subsampled/len(losses[::int(ratio)])}\n{guess_cor_ref_subsampled/(guess_cor_ref_subsampled+int(0.1*len(eval_dataset)))}\n{guess_cor_subsampled/(guess_cor_subsampled+int(0.1*len(eval_dataset)))}")
                    
        else:
            print(f"{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
        print("_____")


if __name__ == "__main__":
    main()


#"args": ["--model_name_or_path", "gpt-2", "--tokenizer_name", "gpt-2", "--train_file", "./data-prep/datasets/augmented-train1.txt", "--mia_train_file", "./data-prep/datasets/original-train1.txt", "--validation_file", "./data-prep/datasets/augmented-test1.txt", "mia_validation_file", "./data-prep/datasets/original-test1.txt", "--block_size", "1024", "--output_dir", "ft_gpt2_mia_aug_trained", "--eval_steps", "1000", "--learning_rate", "1e-5", "--do_ref_model", "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", "8", "--num_train_epochs", "3"] 
