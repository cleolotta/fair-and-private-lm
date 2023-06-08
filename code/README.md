# Language Modelling

This folder contains the code to run the causal language modeling and to perform membership inference attack. The modified run_clm.py from Huggingface is taken from: https://github.com/mireshghallah/ft-memorization

## Example of Usage

**run_clm.py** is used to perform causal language modeling with the possibility of adding differential privacy (by setting add_dp) and to use dropout regularization for debiasing by setting --dropout_debias = True.

E.g., train a model with differential privacy with LoRA and perform a membership inference attack:
```angular2html
python ./code/run_clm.py --train_file [] --validation_file [] --model_name_or_path gpt2-medium --tokenizer_name gpt2-medium --do_ref_model --add_dp
```

**run_clm_aug.py** is used to perform causal language modeling with the possibility of adding differential privacy (by setting add_dp) and to modify the training data with counterfactual data augmentation with the default setting --counterfactual_augmentation = "gender"

E.g., train a model with differential privacy with LoRA, CDA as gender-debiasing method and perform a membership inference attack:
```angular2html
python ./code/run_clm_aug.py --train_file [] --validation_file [] --model_name_or_path gpt2-medium --tokenizer_name gpt2-medium --do_ref_model --add_dp
```

