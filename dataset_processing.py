import pickle
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
import os


class TextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        self.examples = [] # contains the final text training instances. Instances are blocks of length block_size
        all_lines = []
        # open text file
        with open(file_path, "r", encoding="utf-8") as f:
            block = []
            counter = 0
            # read file line by line
            for line in f:
                # status update
                counter += 1
                if (counter % 10000000) == 0:
                    print("Sentences: ", counter)

                # concat lines to block until block has more than 100000 words
                # 100000 because high number means that only a small amount of words is lost when the block of 100000 is cut to smaller blocks of block_size
                # (last words of of the 100000 are lost because usually: 100000 % block_size != 0)
                block = block + line.split()
                number_words = len(block)
                if number_words > 100000:
                    block_text = " ".join(block)
                    block = []
                    # convert words of block to tokens and create smaller blocks of length block_size
                    tokenized_block = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(block_text))
                    for b in range(0, len(tokenized_block) - block_size + 1, block_size):
                         # add the blocks of length block_size to self.examples
                         self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_block[b : b + block_size]))

        print("Number of training instances (Number of blocks of length", block_size, "):", len(self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

