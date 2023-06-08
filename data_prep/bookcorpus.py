#Adapted from the code of Lauscher et al. (2021) Sustainable modular debiasing of language models (https://aclanthology.org/2021.findings-emnlp.411/)

from datasets import load_dataset
import re
import argparse
from pathlib import Path
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="The output for the counterfactual augmented dataset for the book corpus.")
    parser.add_argument("--skip_sentences",
                        type=int,
                        required=True,
                        help="To get just a fraction of the bookcorpus dataset: alternating 'skip X sentences' and 'take Y sentence'")
    parser.add_argument("--block_size",
                        type=int,
                        required=True,
                        help="number of sentences treated as a block: sum of 'skip X senteces' and 'take Y sentences'")
    args = parser.parse_args()


    # Load datasets of wikipedia and bookcorpus
    print("Load bookcorpus...")
    dataset_bookcorpus = load_dataset("bookcorpus")
    print("...done\n\n")


    # create the output text file
    f = open(args.output_file,"w+") #augmented 
    f.close()

    # open the output text file to append sentence by sentence
    with open(args.output_file,'a+', encoding='utf-8') as f:

# ---------------------------------------------BOOKCORPUS---------------------------------------------
        print("Create list of sentences of bookcorpus...")
        counter = 0
        sent_count = 0

        step = 0
        # bookcorpus: extract data block by block
        for block in dataset_bookcorpus["train"]:
            block = block["text"]
            # put spaces in front of signs and replace new lines by dots
            block = block.replace(",", " ,").replace(":", " :").replace(";", " ;").replace("\n", " . ")
            if (counter % 10000000) == 0: # status update
                print("bookcorpus: ", counter, " / ", len(dataset_bookcorpus["train"]))
                print(f"sentence_count_og {sent_count}")


             #split block to list of sentences
            s_list = re.split('[.?!]', block)
            counter += 1

            # alternating skip X (skip_sentences) sentences and take Y (take_sentences) sentences
            for i in range(len(s_list)): # for each single sentence
                s = s_list[i].strip().lower()
                if len(s.split()) < 4: # skip sentences with less than 4 tokens
                    continue
                if step < args.skip_sentences: # skip first X sentences of block
                    step += 1
                    continue
                elif step > args.block_size: # start new block
                    step = 0
                    continue
                else: # take last Y sentences of block
                    step += 1

                # only executed if sentence belongs to last Y sentences of block
                edit = False
                # split sentence to words and eliminate whitespaces around the words
                s_words = s.split()
                s_words[:] = [x.strip() for x in s_words if x]
                
                if len(s_words) < 4:
                    continue
                s_words_og = s_words.copy()
                f.write(" ".join(s_words) + " . \n")
                sent_count += 1

        print("...done\n\n")
        print(f"sentence_count_og {sent_count}")



