Code taken and modified from:
https://github.com/mireshghallah/ft-memorization

**run_clm.py** is used to perform causal language modeling with the possibility of adding differential privacy (by setting add_dp) and to use dropout regularization for debiasing by setting --dropout_debias = True.

**run_clm_aug.py** is used to perform causal language modeling with the possibility of adding differential privacy (by setting add_dp) and to modify the data with counterfactual data augmentation by setting --counterfactual_augmentation = "gender"

