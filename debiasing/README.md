

1. run run_mlm_adapter_v2.py to debias the transformer model


Evaluation with BEC-Pro Evaluation Framework:
1. Run "bias_becpro.py" - output_dir has to contain an "evaluation" folder 
Evaluation with DiSCo Framework:
1. Run "bias_disco.py" - output_dir has to contain an "evaluation" folder
Evaluation with WEAT-Test:
1. Run extract_BERT_embeddings.py to get embeddings
2. Run xweat.py 