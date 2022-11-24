import argparse
import json
import os
import pandas as pd
from collections import defaultdict
import torch
import transformers
from transformers import GPT2Model
import  sys
sys.path.append('C:/Users/cmatz/master-thesis/fair-and-private-lm/bias-bench-main')


#import bias-bench-main
from bias_bench.model import models
from bias_bench.benchmark.seat import SEATRunner
import dp_transformers
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs SEAT benchmark for debiased models.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--tests",
    action="store",
    nargs="*",
    help="List of SEAT tests to run. Test files should be in `data_dir` and have "
    "corresponding names with extension .jsonl.",
)
parser.add_argument(
    "--n_samples",
    action="store",
    type=int,
    default=100000,
    help="Number of permutation test samples used when estimating p-values "
    "(exact test is used if there are fewer than this many permutations).",
)
parser.add_argument(
    "--parametric",
    action="store_true",
    help="Use parametric test (normal assumption) to compute p-values.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="SentenceDebiasBertModel",
    choices=[
        "CDAGPT2Model",
        "DropoutGPT2Model",
        "DPGPT2Model",
        "DPLoRAGPT2Model",
        "LoRAGPT2Model",        
    ],
    help="Debiased model (e.g., SentenceDebiasModel) to evaluate.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="gpt-medium",
    choices=["bert-base-uncased", "albert-base-v2", "roberta-base", "gpt2", "gpt2-medium"],
    help="HuggingFace model name or path (e.g., bert-base-uncased). Checkpoint from which a "
    "model is instantiated.",
)
parser.add_argument(
    "--load_path",
    action="store",
    type=str,
    help="Path to saved CDA or Dropout model checkpoint.",
)
parser.add_argument(
    "--bias_type",
    action="store",
    type=str,
    choices=["gender", "religion", "race"],
    help="The type of bias to mitigate.",
)
parser.add_argument(
    "--objective",
    type=str,
    default=None,
)
parser.add_argument("--lora_dim", default=4, type=int,  help= "LoRA dimension; 0 means LoRA is disabled")
parser.add_argument("--lora_dropout", default=0.0, type=float,  help= "Dropout probability for LoRA layers")
parser.add_argument('--lora_alpha', default=32, type=int,  help="LoRA attention alpha")


if __name__ == "__main__":
    args = parser.parse_args()

    print("Running SEAT benchmark:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - tests: {args.tests}")
    print(f" - n_samples: {args.n_samples}")
    print(f" - parametric: {args.parametric}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - load_path: {args.load_path}")
    print(f" - bias_type: {args.bias_type}")




    if args.model == "DPLoRAGPT2Model":
        model = getattr(models, args.model)(
        args.model_name_or_path,args.load_path, args.lora_dim, args.lora_alpha, args.lora_dropout)
        print("Param: h.21.attn.c_attn.lora_A.weight")
        print(model.state_dict()["h.21.attn.c_attn.lora_A.weight"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)
    elif args.model =="LoRAGPT2Model":
        model = getattr(models, args.model)(
            args.model_name_or_path, args.load_path, args.lora_dim, args.lora_alpha, args.lora_dropout
        )
        print("Param: h.21.attn.c_attn.lora_A.weight")
        print(model.state_dict()["h.21.attn.c_attn.lora_A.weight"])
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)
    elif args.load_path is None: # load pre-trained huggingface model
        model = getattr(models, args.model)(
        args.model_name_or_path
        )
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    else: # load checkpoint of model without LoRA 
        model = getattr(models, args.model)(
            args.load_path
        )
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)


    runner = SEATRunner(
        tests=args.tests,
        data_dir=f"{args.persistent_dir}/data/seat",
        n_samples=args.n_samples,
        parametric=args.parametric,
        model=model,
        tokenizer=tokenizer,
    )
    results = runner()
    print(results)

    os.makedirs(f"{args.persistent_dir}/results/seat", exist_ok=True)
    with open(f"{args.persistent_dir}/results/seat/{args.objective}_model", "w") as f:
        json.dump(results, f)
