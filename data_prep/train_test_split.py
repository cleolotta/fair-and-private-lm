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


    df = pd.read_table(args.input_file, names=['text'])
    print("read into df")
    length = len(df)

    print(length)
    ind = int(0.8 * length)
    training_data = df[:ind]
    testing_data = df[ind:]
       
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
    
