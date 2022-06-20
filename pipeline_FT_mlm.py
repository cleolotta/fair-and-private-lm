import torch
from torch import cuda
from torch.utils.data.dataset import Dataset
import logging
import os
from typing import Dict
from transformers import (
    EvalPrediction,
    GlueDataset,
    GlueDataTrainingArguments,
    AdapterType,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
    glue_tasks_num_labels,
    TrainingArguments,
)

from arguments import ModelArguments, DataTrainingArguments, LanguageAdapterArguments, TaskAdapterArguments, FinetuningArguments
from dataset_processing import TextDataset
from bias_becpro import evaluate_becpro
from bias_disco import evaluate_adjusted_disco


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in arguments.py and src/transformers/training_args.py
    # Parse the arguments and create data classes
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LanguageAdapterArguments, TaskAdapterArguments, FinetuningArguments, TrainingArguments))
    model_args, data_args, la_args, ta_args, ft_args, training_args = parser.parse_args_into_dataclasses()

    # Check for argument conflicts
    if (data_args.language_eval_data_file is None and ft_args.eval_finetuning):
        raise ValueError("Cannot do evaluation without an evaluation data file.")
    if (ft_args.do_finetuning and data_args.language_train_data_file is None):
        raise ValueError("Cannot do training without an training data file.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    print()
    logger.info(model_args)
    logger.info(data_args)
    logger.info(ft_args)

    # Create result folders
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
        os.makedirs("{0}/evaluations".format(training_args.output_dir))
        if ft_args.do_finetuning:
            os.makedirs("{0}/finetuned_model".format(training_args.output_dir))
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

    # Adjust block_size if too low or too high
    if la_args.train_language_adapter:
        if data_args.block_size <= 0:
            data_args.block_size = 128
        else:
            data_args.block_size = min(data_args.block_size, tokenizer.max_len)


    # Create model config
    print("\n\nCreate model config...")
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("No valid model_name_or_path.")
    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm flag.")
    print("...done")

    # Create model with language model head and specified model config
    print("\n\nCreate model with language model head...")
    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        raise ValueError("No valid model_name_or_path.")
    model.resize_token_embeddings(len(tokenizer))
    print("...done")


    print("\n\n\n-----------------------------------------------\n------------------Finetuning-------------------\n-----------------------------------------------")

    # If finetuning is desired
    if ft_args.do_finetuning:  
        # Get datasets for finetuning
        print("\n\nGet train dataset for finetuning...")
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.language_train_data_file, block_size=data_args.block_size)
        print("...done\nGet eval dataset for finetuning...")
        eval_dataset = TextDataset(tokenizer=tokenizer, file_path=data_args.language_eval_data_file, block_size=data_args.block_size)
        print("...done")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)

        # Update training_args with ft_args to create Trainer
        training_args.do_train = ft_args.do_finetuning
        training_args.do_eval = ft_args.eval_finetuning
        training_args.per_device_train_batch_size = ft_args.finetuning_batch_size
        training_args.per_device_eval_batch_size = ft_args.finetuning_batch_size
        training_args.learning_rate = ft_args.finetuning_learning_rate
        training_args.num_train_epochs = ft_args.finetuning_epochs
        # Initialize Trainer
        print("\n\nDo finetuning...")
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
        )
        # train model 
        trainer.train()
        print("...done")

        # save model to output folder and to finetuned models folder
        model.save_pretrained('{0}/finetuned_model/'.format(training_args.output_dir))
        if not os.path.exists("./finetuned_models/{0}".format(training_args.output_dir[10:])):
            os.makedirs("./finetuned_models/{0}".format(training_args.output_dir[10:]))
        model.save_pretrained('./finetuned_models/{0}'.format(training_args.output_dir[10:]))
        print("...done")

        # Evaluation
        if training_args.do_eval:
            print("\n\nDo finetuning evaluation...")
            logger.info("*** Evaluate ***")
            # Evaluate finetuning and save the loss to output file
            eval_output = trainer.evaluate()
            result = {"loss": eval_output["eval_loss"]}
            output_eval_file = os.path.join("{0}/evaluations/".format(training_args.output_dir), "evaluation_results_finetuning.txt")
            if trainer.is_world_master():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
        print("...done")

    else:
        print("\n\nFinetuning is skipped because already trained model is loaded.\n\n")


    print("\n\n\n-----------------------------------------------\n----------------Bias Evaluation----------------\n-----------------------------------------------\n\n")

    # Bias evaluation if desired
    if data_args.bias_eval_task == "disco":
        # Evaluate on DisCo
        print("Evaluate on DisCo ...")
        evaluate_adjusted_disco(model.cuda(), tokenizer, training_args.output_dir, la_args.language_adapter)
    elif data_args.bias_eval_task == "bec-pro_english":
        # Evaluate on BEC-Pro
        print("Evaluate on English BEC-Pro ...")
        evaluate_becpro("english", model.cuda(), tokenizer, training_args.output_dir, la_args.language_adapter)
    elif data_args.bias_eval_task == "bec-pro_german":
        # Evaluate on BEC-Pro
        print("Evaluate on English BEC-Pro ...")
        evaluate_becpro("german", model.cuda(), tokenizer, training_args.output_dir, la_args.language_adapter)
    else:
        print("\n\n\nNo bias evaluation wished...")



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
