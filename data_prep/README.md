Code taken and modified from:
https://aclanthology.org/2021.findings-emnlp.411/

**CDA_bookcorpus** is used to create a subset of the bookcorpus. Decide of how many sentences a block should contain and how many of those sentences to skip in each block.
We used: python data_prep/CDA_bookcorpus.py --output_file "./data_prep/data.txt" --skip_sentences 120 --block_size 128

**cda_train_test_split.py** is used to separate the dataset into train (80%) and test (20%) data.

**cda_words.py** is used during the counterfactual data augmentation process in code/run_clm_aug.py

