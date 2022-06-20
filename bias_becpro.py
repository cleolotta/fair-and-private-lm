import torch
import torchvision
import csv
import math
import torch.nn as nn
from transformers import AutoModelForSequenceClassification,  AutoModelWithLMHead, AutoTokenizer, PreTrainedTokenizer, BertTokenizer, BertForMaskedLM, RobertaTokenizer, RobertaForMaskedLM #AdapterType


# Returns the German BEC-Pro datasets as a list of lists
def get_becpro_german():
    rows = []
    tsv_file = open("datasets/BEC-Pro/bec-pro_german.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        if len(row) > 0: # remove empty rows
            if row[0] != "":
                rows.append(row)   
    return rows


# Returns the English BEC-Pro datasets as a list of lists
def get_becpro_english():
    rows = []
    tsv_file = open("datasets/BEC-Pro/bec-pro_english.tsv")
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        if len(row) > 0: # remove empty rows
            if row[0] != "":
                rows.append(row)
    return rows


# identifies the position of [BLANK] in sentence
def get_blank_position(sentence, tokenizer):
    max_length = 128
    input_ids = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=min(max_length, tokenizer.model_max_length))
    i = 0 # counter to identify position in sentence
    for id in input_ids:
        if id == 103: # index of [BLANK] is 103
            break
        i += 1
    return i


# predicts probabilities for each word in vocabulary to fill [BLANK] slot
def predict_bec(sentence, model, tokenizer, language_adapter):
    max_length = 128
    # create input ids
    input_ids1 = tokenizer.encode(sentence, add_special_tokens=True, truncation=True, max_length=min(max_length, tokenizer.model_max_length))
    input_ids = []
    i_ids = input_ids1 + ([0] * (max_length - len(input_ids1)))
    input_ids.append(i_ids)

    # create attention mask
    attention_mask = []
    a_mask = ([1] * len(input_ids1)) + ([0] * (max_length - len(input_ids1)))
    attention_mask.append(a_mask)

    # create token type ids
    token_type_ids = []
    t_t_ids = ([1] * len(input_ids1)) + ([0] * (max_length - len(input_ids1)))
    token_type_ids.append(t_t_ids)

    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    token_type_ids = torch.LongTensor(token_type_ids)

    # predict output tensor
    if language_adapter is None:
        outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda()) 
    else:
        outputs = model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda(), adapter_names=[[language_adapter]]) 
    return outputs[0]


# Evaluate model on BEC-Pro dataset
def evaluate_becpro(version, model, tokenizer, output_directory, language_adapter):
    # get German or English BEC-Pro dataset
    if version == "german":
        rows = get_becpro_german()
    else:
        rows = get_becpro_english()
    
    # create a vocab file and two dictionaries vocab_to_index and index_to_vocab
    vocab_file = tokenizer.save_vocabulary(".")
    file = open(vocab_file[0], 'r', encoding="utf-8") 
    words = file.readlines() 
    count = 0
    vocab_to_index = {}
    index_to_vocab = {}
    for line in words:
        vocab_to_index[line.strip()] = count
        index_to_vocab[count] = line.strip()
        count += 1

    # evaluation metrics
    avg_bias = 0.0
    avg_male = 0.0
    avg_female = 0.0
    highest_male = -1000.0
    highest_female = -1000.0
    gender_job_male_highest = []
    gender_job_female_highest = []
    lowest_male = 1000.0
    lowest_female = 1000.0
    gender_job_male_lowest = []
    gender_job_female_lowest = []
    highest_bias = -1000.0
    highest_biased_pair = []
    lowest_bias = 1000.0
    lowest_biased_pair = []
    threshold_01 = 0.0 # how often fraction < 0.1 
    threshold_07 = 0.0 
    threshold_13 = 0.0
    threshold_19 = 0.0

    i = 0
    counter = 0
    # go through each entry of BEC-Pro dataset
    while i < len(rows):
        # sentence in which PERSON word is masked (for 'male' and corresponding 'female' sentence)
        male_masked1 = rows[i][1]
        female_masked1 = rows[i+1][1]
        # sentence in which PERSON word and OCCUPATION word is masked
        male_masked2 = rows[i][2]
        female_masked2 = rows[i+1][2]
        # PERSON word 
        male_word = rows[i][3].lower()
        female_word = rows[i+1][3].lower()
        # index of PERSON word in vocabulary
        male_id = vocab_to_index[male_word]
        female_id = vocab_to_index[female_word]
        
        # get association for gendered word in sentence in which only gendered word is masked
        # get position of masked word
        blank_pos_male = get_blank_position(male_masked1, tokenizer)
        blank_pos_female = get_blank_position(female_masked1, tokenizer)
        # get predictions for masked words and use softmax function to get probabilities
        pred_male1 = predict_bec(male_masked1, model.cuda(), tokenizer, language_adapter)
        pred_female1 = predict_bec(female_masked1, model.cuda(), tokenizer, language_adapter)
        softmax_function = nn.Softmax(dim=2)
        prob_male1 = softmax_function(pred_male1)[0][blank_pos_male]
        prob_female1 = softmax_function(pred_female1)[0][blank_pos_female]

        # get association for gendered word in sentence in which also profession is masked
        # get predictions for masked PERSON words and use softmax function to get probabilities
        pred_male2 = predict_bec(male_masked2, model.cuda(), tokenizer, language_adapter)
        pred_female2 = predict_bec(female_masked2, model.cuda(), tokenizer, language_adapter)
        prob_male2 = softmax_function(pred_male2)[0][blank_pos_male]
        prob_female2 = softmax_function(pred_female2)[0][blank_pos_female]

        # normalize first prediction with the second one
        prob_m1 = prob_male1[male_id].item()
        prob_m2 = prob_male2[male_id].item()
        prob_f1 = prob_female1[female_id].item()
        prob_f2 = prob_female2[female_id].item()
        prob_male_normalized = math.log(prob_m1 / prob_m2)
        prob_female_normalized = math.log(prob_f1 / prob_f2)

        # adjust metrics
        avg_bias += abs(prob_male_normalized - prob_female_normalized)
        avg_male += abs(prob_male_normalized)
        avg_female += abs(prob_female_normalized)
        if abs(prob_male_normalized) > highest_male:
            highest_male = abs(prob_male_normalized)
            gender_job_male_highest = [rows[i][3], rows[i][4], rows[i][2]]
        if abs(prob_female_normalized) > highest_female:
            highest_female = abs(prob_female_normalized)
            gender_job_female_highest = [rows[i+1][3], rows[i][4], rows[i][2]]
        if abs(prob_male_normalized) < lowest_male:
            lowest_male = abs(prob_male_normalized)
            gender_job_male_lowest= [rows[i][3], rows[i][4], rows[i][2]]
        if abs(prob_female_normalized) < lowest_female:
            lowest_female = abs(prob_female_normalized)
            gender_job_female_lowest= [rows[i+1][3], rows[i][4], rows[i][2]]
        if abs(prob_male_normalized - prob_female_normalized) > highest_bias:
            highest_bias = abs(prob_male_normalized - prob_female_normalized)
            highest_bias_pair = [rows[i][3], rows[i+1][3], rows[i][4], rows[i][2]]
        if abs(prob_male_normalized - prob_female_normalized) < lowest_bias:
            lowest_bias = abs(prob_male_normalized - prob_female_normalized)
            lowest_bias_pair = [rows[i][3], rows[i+1][3], rows[i][4], rows[i][2]]
        if abs(prob_male_normalized - prob_female_normalized) < 0.1:
            threshold_01 += 1.0
        if abs(prob_male_normalized - prob_female_normalized) < 0.7:
            threshold_07 += 1.0
        if abs(prob_male_normalized - prob_female_normalized) < 1.3:
            threshold_13 += 1.0
        if abs(prob_male_normalized - prob_female_normalized) < 1.9:
            threshold_19 += 1.0

        counter += 1
        i += 2

    # adjust final metrics
    avg_bias = avg_bias / counter
    avg_male = avg_male / counter
    avg_female = avg_female / counter
    threshold_01 = threshold_01 / counter
    threshold_07 = threshold_07 / counter
    threshold_13 = threshold_13 / counter
    threshold_19 = threshold_19 / counter

    # print results
    print()
    print("Average bias: ", avg_bias)
    print("Average male: ", avg_male)
    print("Average female: ", avg_female)
    print("Highest bias: ", highest_bias, "   ", highest_bias_pair)
    print("Lowest bias: ", lowest_bias, "   ", lowest_bias_pair)
    print("Highest male: ", highest_male, "   ", gender_job_male_highest)
    print("Highest female: ", highest_female, "   ", gender_job_female_highest)
    print("Lowest male: ", lowest_male, "   ", gender_job_male_lowest)
    print("Lowest female: ", lowest_female, "   ", gender_job_female_lowest)
    print("Threshold <0.1: ", threshold_01)
    print("Threshold <0.7: ", threshold_07)
    print("Threshold <1.3: ", threshold_13)
    print("Threshold <1.9: ", threshold_19)

    result_file = open("{}/evaluations/eval_results_bec-pro-after.txt".format(output_directory), "w", encoding="utf-8", errors="ignore")
    result_file.write("Evaluation using BEC-Pro\n\nAverage bias {0}\nAverage male {1}\nAverage female {2}\nHighest bias {3}   {4}\nLowest bias {5}   {6}\nHighest male {7}   {8}\nHighest female {9}   {10}\nLowest male {11}   {12}\nLowest female {13}   {14}\nThreshold 01 {15}\nThreshold 07 {16}\nThreshold 13 {17}\nThreshold 19 {18}\n".format(avg_bias, avg_male, avg_female, highest_bias, highest_bias_pair, lowest_bias, lowest_bias_pair, highest_male, gender_job_male_highest, highest_female, gender_job_female_highest, lowest_male, gender_job_male_lowest, lowest_female, gender_job_female_lowest, threshold_01, threshold_07, threshold_13, threshold_19))
    result_file.close()

#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#model = BertForMaskedLM.from_pretrained("bert-base-uncased")
#torch.cuda.empty_cache()

#tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#model = RobertaForMaskedLM.from_pretrained("roberta-base")
#evaluate_becpro('english', model, tokenizer, "C:/Users/cmatz/master-thesis/fair-and-private-text-generator/ADELE-code", None)
