import torch
from torch import cuda
from torch.utils.data.dataset import Dataset
import logging
import os
from typing import Dict
import numpy as np
from transformers import (
    EvalPrediction,
    AdapterType,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
    glue_tasks_num_labels,
    TrainingArguments,
    BertForMaskedLM,
    BertModel,
    BertTokenizer
)


from arguments import ModelArguments, DataTrainingArguments, LanguageAdapterArguments, TaskAdapterArguments, FinetuningArguments
from dataset_processing import TextDataset


logger = logging.getLogger(__name__)



# get vocabulary: english or multilingual
def get_vocab(language):
    vocab = []
    if language == "en":
        file = open("XWEAT/data/vocab_en.txt", 'r', encoding='utf-8')
        words = file.readlines()
        for word in words:
           vocab.append(word)
    else:
        file = open("XWEAT/data/multilingual_vocab.txt", 'r', encoding='utf-8')
        words = file.readlines()
        for word in words:
           vocab.append(word)
    return vocab
        

# run model to get embeddings of in_text
def tokenize_and_encode_text(in_text, tokenizer, model, model_type, language_adapter):
    # Specify text
    text = "[CLS] " + in_text + " [SEP]"
    # Initial tokenization
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = len(indexed_tokens)*[0]
    # Convert inputs to Pytorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Predict hidden states features for each layer
    with torch.no_grad():
        # run model with or without language adapter and get hidden layers
        if language_adapter is not None:
            outputs = model(tokens_tensor, token_type_ids=segments_tensors, adapter_names=[[language_adapter]])
            # All hidden layers are captured here
            hlayers = outputs[1]
            _, seq_len, dim = hlayers[0].shape
        else:
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            # All hidden layers are captured here
            last_layer = outputs[0]
            _, seq_len, dim = last_layer.shape
            hlayers = outputs[2]

        # Go over all subwords
        final_avg_vector = np.zeros(dim)
        for i in range(1,seq_len-1):
            # Go over all hidden layers
            avg_vector = np.zeros(dim)
            # SET HERE HOW MANY AND WHICH HIDDEN LAYERS TO USE: hlayers[X:Y]
            for hlayer in hlayers[:]:
                avg_vector += np.array(hlayer[0][i])
            avg_vector = avg_vector / np.linalg.norm(avg_vector)
            final_avg_vector += np.array(avg_vector)
        final_avg_vector = final_avg_vector / np.linalg.norm(final_avg_vector)
        final_vector = [round(elem, 5) for elem in final_avg_vector]
        in_text1 = str(in_text).strip().replace(" ", "_")
        return in_text1, final_vector


def main():
    # See all possible arguments in arguments.py and src/transformers/training_args.py
    # Parse the arguments and create data classes
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LanguageAdapterArguments, TaskAdapterArguments, FinetuningArguments, TrainingArguments))
    model_args, data_args, la_args, ta_args, ft_args, training_args = parser.parse_args_into_dataclasses()

    # Check for argument conflicts
    if (la_args.load_language_adapter is None) and (la_args.language_adapter is not None):
        raise ValueError("Language adapter must be loaded. With this file, language adapters cannot be created and trained from scratch. Therefore, use f.e. pipeline_LA.py")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    print()
    logger.info(model_args)
    logger.info(data_args)
    logger.info(la_args)
    # Set seed
    set_seed(training_args.seed)

    # Load tokenizer
    print("\n\nLoad tokenizer...")
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("You need to specify which tokenizer to be loaded. Give a valid tokenizer_name or model_name_or_path.")
    print("...done")

    # Create BERT model
    print("\n\n\nCreate BERT model...")
    if la_args.language_adapter is not None:
        model = BertForMaskedLM.from_pretrained(model_args.model_name_or_path, output_hidden_states=True)
        # Load language adapter
        print("\n\nLoad language adapter...")
        model.load_adapter("./language_adapters/{0}".format(la_args.language_adapter))
        model.set_active_adapters(la_args.language_adapter)
    else:
        model = BertModel.from_pretrained(model_args.model_name_or_path, output_hidden_states=True)
    model.eval()
    print("...done")


    # Create the embeddings and write them to output_file
    print("\n\n\nCreate embeddings...")
    vocab = get_vocab("multilingual") # get vocabulary of bert-base-multilingual-uncased version
    with open(training_args.output_dir, "w+", encoding='utf-8') as out:
        for text in vocab:
            in_text1, final_vector = tokenize_and_encode_text(text, tokenizer, model, model_args.model_type, la_args.language_adapter)
            out.write(in_text1 + " ")
            out.write(" ".join([str(val) for val in final_vector]))
            out.write("\n")
    print("...done")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
