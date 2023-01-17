import argparse
import json
import os

import torch
import transformers
import sys
sys.path.append('./bias-evaluation')


from bias_bench.benchmark.stereoset import StereoSetRunner
from bias_bench.model import models
import dp_transformers
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _is_generative(model):
    # Checks if we are running an autoregressive model.
    return model in [
        "GPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "DPGPT2LMHeadModel",
        "LoRAGPT2LMHeadModel",
        "DPLoRAGPT2LMHeadModel"
    ]


thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs StereoSet benchmark.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data will be stored.",
)
parser.add_argument(
    "--model",
    action="store",
    type=str,
    default="SentenceDebiasForMaskedLM",
    choices=[
        "GPT2LMHeadModel",
        "CDAGPT2LMHeadModel",
        "DropoutGPT2LMHeadModel",
        "DPGPT2LMHeadModel",
        "DPLoRAGPT2LMHeadModel",
        "LoRAGPT2LMHeadModel",
        "LoRAptGPT2LMHeadModel",
    ],
    help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM).",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="gpt2-medium",
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
    "--batch_size",
    action="store",
    type=int,
    default=1,
    help="The batch size to use during StereoSet intrasentence evaluation.",
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

    print("Running StereoSet:")
    print(f" - persistent_dir: {args.persistent_dir}")
    print(f" - model: {args.model}")
    print(f" - model_name_or_path: {args.model_name_or_path}")
    print(f" - load_path: {args.load_path}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - bias_type: {args.bias_type}")


    if args.model == "DPLoRAGPT2LMHeadModel": # load checkpoint of dp model with LoRA
        model = getattr(models, args.model)(
        args.model_name_or_path, args.load_path, args.lora_dim, args.lora_alpha, args.lora_dropout)
        print("Layer: transformer.h.21.attn.c_attn.lora_A.weight")
        print(model.state_dict()["transformer.h.21.attn.c_attn.lora_A.weight"])
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)
    elif args.model =="LoRAGPT2LMHeadModel": # load checkpoint of model with LoRA
        model = getattr(models, args.model)(
        args.model_name_or_path,args.load_path, args.lora_dim, args.lora_alpha, args.lora_dropout)
        print("Layer: transformer.h.21.attn.c_attn.lora_A.weight")
        print(model.state_dict()["transformer.h.21.attn.c_attn.lora_A.weight"])
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)
    elif args.load_path is None: # load pre-trained huggingface model
        model = getattr(models, args.model)(
        args.model_name_or_path, args.lora_dim, args.lora_alpha, args.lora_dropout
        )
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    else: # load checkpoint of model without LoRA 
        model = getattr(models, args.model)(
            args.load_path
        )
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)


    # Use self-debiasing name.
    bias_type = args.bias_type
    if bias_type == "race":
        bias_type = "race-color"

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"{args.persistent_dir}/data/stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=_is_generative(args.model),
        bias_type=bias_type,
    )
    results = runner()


    os.makedirs(f"{args.persistent_dir}/results/stereoset", exist_ok=True)
    with open(
        f"{args.persistent_dir}/results/stereoset/{args.objective}_model.json", "w"
    ) as f:
        json.dump(results, f, indent=2)
