# https://gist.github.com/yuchenlin/eb63e2d0513f70cfc9bb85fa5a78953b
import numpy as np 
import csv
import os
import argparse
import tqdm 
from tqdm import tqdm
import transformers
import torch
import pandas as pd
import sys
sys.path.append('C:/Users/cmatz/master-thesis/fair-and-private-lm/bias-bench-main')
from bias_bench.model import models
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import dp_transformers
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora
from scipy.special import softmax
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = torch.device("cuda:0" if cuda else "cpu")
SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
np.random.seed(SEED)

def get_becpro_english():
        rows = []
        tsv_file = open("./bias-bench-main/data/bec-pro/bec-pro_english.tsv")
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            if len(row) > 0: # remove empty rows
                if row[0] != "":
                    rows.append(row)
        return rows
thisdir = os.path.dirname(os.path.realpath(__file__))

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent_dir",
                        action="store",
                        type=str,
                        default=os.path.realpath(os.path.join(thisdir, "..")),
                        help="Directory where all persistent data will be stored.",)
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=False,
                        default="gpt2-medium",)
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
        ],
        help="Model to evalute (e.g., SentenceDebiasBertForMaskedLM).",
    )
    parser.add_argument(
        "--load_path",
        action="store",
        default=None,
        type=str,
        help="Path to saved CDA or Dropout model checkpoint.",
    )
    parser.add_argument(
    "--objective",
    type=str,
    default=None,
    )
    parser.add_argument("--lora_dim", default=4, type=int,  help= "LoRA dimension; 0 means LoRA is disabled")
    parser.add_argument("--lora_dropout", default=0.0, type=float,  help= "Dropout probability for LoRA layers")
    parser.add_argument('--lora_alpha', default=32, type=int,  help="LoRA attention alpha")
    args = parser.parse_args()  
    

    kwargs = {}

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    #model = getattr(models, args.model)(
    #    args.load_path or args.model_name_or_path, **kwargs
    #)
    
    if args.model == "DPLoRAGPT2LMHeadModel": # load checkpoint of dp model with LoRA
        model = getattr(models, args.model)(
        args.model_name_or_path, args.load_path, args.lora_dim, args.lora_alpha, args.lora_dropout)
        print("Layer: transformer.h.21.attn.c_attn.lora_A.weight")
        print(model.state_dict()["transformer.h.21.attn.c_attn.lora_A.weight"])
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.load_path)
    elif args.model =="LoRAGPT2LMHeadModel": # load checkpoint of model with LoRA
        model = getattr(models, args.model)(
            args.model_name_or_path, args.load_path, args.lora_dim, args.lora_alpha, args.lora_dropout
        )
        print("Layer: transformer.h.21.attn.c_attn.lora_A.weight")
        print(model.state_dict()["transformer.h.21.attn.c_attn.lora_A.weight"])
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
    
    
    model.to(device)
    
    def sent_scoring(model, tokenizer, text, cuda):
        assert model is not None
        assert tokenizer is not None
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]
        sentence_prob = loss.item()
        return sentence_prob
        
    rows = get_becpro_english()
    i=0
    counter = 0
    male_higher = 0
    max_diff = 0
    max_index = 0
    max_fem = 0
    max_mal = 0

    female_ = list()
    with open(f"{args.persistent_dir}/results/bec-pro/{args.objective}_model.json", "w") as result_file:
        pbar = tqdm(total = len(rows))
        while i < len(rows):
            male_masked = rows[i][0]
            female_masked = rows[i+1][0]
            prob_male = sent_scoring(model,tokenizer, male_masked, cuda=False)
            prob_female = sent_scoring(model,tokenizer, female_masked, cuda=False)

            if prob_male < prob_female:
                male_higher += 1
            diff = abs(prob_male - prob_female)
            if diff > max_diff:
                max_diff = diff
                max_index = i
                max_fem = prob_female
                max_mal = prob_male
            pbar.update(2)
            i +=2
            if i%2==0:
                counter += 1


        avg_bias = male_higher/(counter)*100
        result_file.write(f"\n \n \n Evaluation Average bias {avg_bias}")
        print(f'average bias: {avg_bias}')
        result_file.write("\n Most difference in prediction between male and female word \n MALE MASKED: {0}, SCORE: {1}".format(rows[max_index][0], max_mal))
        result_file.write("\n FEMALE MASKED: {0}, SCORE: {1}".format(rows[max_index+1][0], max_fem))


if __name__ == '__main__':
    main()

    
# python ./bias-bench-main/experiments/bias-becpro-gpt2.py --model_name_or_path "C:\Users\cmatz\master-thesis\fplm\models\ft_gpt2-medium_dp_5epochs\model\best" --tokenizer_name_or_path "C:\Users\cmatz\master-thesis\fplm\models\ft_gpt2-medium_dp_5epochs\model" --output_file "./bias-bench-main/results/bec-pro/bec-pro_bias_gpt2-medium-ft-dp_5epochs_v2.txt"
#tokenizer = GPT2Tokenizer.from_pretrained("C:/Users/cmatz/master-thesis/fplm/models/ft_gpt2-medium_dp_10epochs/model")
#model = GPT2LMHeadModel.from_pretrained("C:/Users/cmatz/master-thesis/fplm\models/ft_gpt2-medium_dp_10epochs/model/best")
#prob_male = sent_scoring((model,tokenizer), "the nurse is a man", cuda=True)
#prob_femal = sent_scoring((model,tokenizer), "the nurse is a woman", cuda=True)