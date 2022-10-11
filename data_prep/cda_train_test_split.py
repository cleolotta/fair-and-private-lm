# this code comes from: Fair and Argumentative Language Modeling for Computational Argumentation by Carolin Holtermann, Anne Lauscher, Simone Ponzetto

import re
import random
import nltk
import itertools
from itertools import *
from nltk import tokenize
import pandas as pd
from tqdm import *
import argparse
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Input file directory")
    parser.add_argument("--output_test",
                        type=str,
                        required=True,
                        help="The output test file")
    parser.add_argument("--output_train",
                        type=str,
                        required=True,
                        help="The output train file")
    args = parser.parse_args()

    #dataset = load_dataset("text", data_files=args.input_file, split='train', cache_dir='cda_cache')

    #dataset = dataset.train_test_split(test_size=0.2, shuffle= False)
    print('hi')
    df = pd.read_table(args.input_file, names=['text'])
    df = df[:500]
    print(df['text'][0])
    print("read into df")
    #print(df[0])
    #txt_a = Path(args.input_file).resolve()
    #length = sum(1 for row in open(txt_a, "r", encoding= "utf-8"))
    length = len(df)

    print(length)
    ind = int(0.8 * length)
    training_data = df[:ind]
    print(training_data['text'][0])
    testing_data = df[ind:]
    
    
    #training_data, testing_data = train_test_split(df, test_size=0.2, shuffle=True)
   
    print(len(training_data))
    print(len(testing_data))


    # Write to file for reconstructability
    with open(args.output_train, 'w', encoding='utf8') as training_file:
        for arg in training_data['text']:
            if len(arg) != 0:
                training_file.write(arg)
                training_file.write('\n')
            else: 
                print("length is 0")

    with open(args.output_test, 'w', encoding='utf8') as test_file:
        for arg in testing_data['text']:
            if len(arg) != 0:
                test_file.write(arg)
                test_file.write('\n')
            else:
                print("length is 0")


if __name__ == "__main__":
    main()
    
# python data_prep/cda_train_test_split.py --input_file "./data_prep/data/original-train.txt" --output_test "./test/original-test.txt" --output_train "./test/original-train.txt"    
# python cda_train_test_split.py --input_file "./datasets/augmented_data.txt" --output_test "./datasets/augmented-test1.txt" --output_train "./datasets/augmented-train1.txt"    
# python data-prep/cda_train_test_split.py --input_file "C:/Users/cmatz/master-thesis/fplm/datasets/original_data.txt" --output_test "pups1" --output_train "pups.txt"