# https://gist.github.com/yuchenlin/eb63e2d0513f70cfc9bb85fa5a78953b
import numpy as np 
import csv
import argparse
import tqdm 
from tqdm import tqdm
import transformers
import torch
import sys
sys.path.append('C:/Users/cmatz/master-thesis/fair-and-private-lm/bias-bench-main')
from bias_bench.model import models
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from scipy.special import softmax
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

def get_becpro_english():
        rows = []
        tsv_file = open("./bias-bench-main/data/bec-pro/bec-pro_english.tsv")
        read_tsv = csv.reader(tsv_file, delimiter="\t")
        for row in read_tsv:
            if len(row) > 0: # remove empty rows
                if row[0] != "":
                    rows.append(row)
        return rows

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        type=str,
                        required=False,
                        default="gpt2",)
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="The output file to store the measured bias")
    parser.add_argument(
        "--model",
        action="store",
        type=str,
        default="SentenceDebiasBertModel",
        choices=[
            "GPT2LMHeadModel",
            "CDAGPT2LMHeadModel",
            "DropoutGPT2LMHeadModel",
            "DPGPT2LMHeadModel"
        ],
        help="Debiased model (e.g., SentenceDebiasModel) to evaluate.",
    )
    parser.add_argument(
        "--load_path",
        action="store",
        type=str,
        help="Path to saved CDA or Dropout model checkpoint.",
    )
    args = parser.parse_args()  

    kwargs = {}

    # Load model and tokenizer. `load_path` can be used to override `model_name_or_path`.
    model = getattr(models, args.model)(
        args.load_path or args.model_name_or_path, **kwargs
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    if cuda:
            model.to(device)
    
    def sent_scoring(model_tokenizer, text, cuda):
        model = model_tokenizer[0]
        tokenizer = model_tokenizer[1]
        assert model is not None
        assert tokenizer is not None
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
        if cuda:
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
    result_file = open(args.output_file, "w")
    result_file.close()
    with open(args.output_file, "w") as result_file:
        while i < len(rows):
            male_masked = rows[i][0]
            female_masked = rows[i+1][0]
            prob_male = sent_scoring((model,tokenizer), male_masked, cuda=True)
            prob_female = sent_scoring((model,tokenizer), female_masked, cuda=True)

            if prob_male < prob_female:
                male_higher += 1
            diff = abs(prob_male - prob_female)
            if diff > max_diff:
                max_diff = diff
                max_index = i
                max_fem = prob_female
                max_mal = prob_male
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