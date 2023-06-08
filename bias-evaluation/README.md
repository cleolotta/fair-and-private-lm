# Bias Evaluation

This folder contains the code to evaluate language models for a gender bias. The introduced frameworks are SEAT (May et al., 2019), StereoSet (Nadeem et al.,) and BEC-Pro (Bartl et al., 2020).

## Example of Usage

##### BEC-Pro
Run BEC-Pro with:

```angular2html
python ./bias-evaluation/experiments/bec-pro.py --model_name_or_path gpt2-medium --load_path "[path_to_model]" --model "DPLoRAGPT2LMHeadModel" --objective "dp_lora"
```

##### StereoSet
Run StereoSet with:

```angular2html
python bias-evaluation/stereoset_debias.py --model_name_or_path "gpt2-medium" --load_path "path_to_model" --model "LoRAGPT2LMHeadModel" --objective "baseline_lora"
```
and evaluate the results with:
```angular2html
python bias-evaluation/experiments/stereoset_evaluation.py --predictions_file "[path_to_prediction_file]" --output_file "[path_to_output_file]"
```

##### SEAT
Run SEAT-tests with:
```angular2html
python bias-evaluation/experiments/seat_debias.py --tests sent-weat6 sent-weat6b sent-weat7 sent-weat7b sent-weat8 sent-weat8b --parametric --model_name_or_path "gpt2-medium" --load_path "[path_to_model]" --model "LoRAGPT2Model" --objective "cda_lora"
```




