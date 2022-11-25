Code taken and slightly modified from:
https://github.com/McGill-NLP/bias-bench/tree/main/bias_bench

@article{meade2021empirical,
  title={An empirical survey of the effectiveness of debiasing techniques for pre-trained language models},
  author={Meade, Nicholas and Poole-Dayan, Elinor and Reddy, Siva},
  journal={arXiv preprint arXiv:2110.08527},
  year={2021}
}

Content that is not from bias_bench repo:
- data - bec-pro
- experiments - bias-becpro-gpt2.py -> origin with adjustments: https://aclanthology.org/2021.findings-emnlp.411/

The evaluations can be executed for example with:

##### BEC-Pro
python ./bias-evaluation/experiments/bec-pro.py --model_name_or_path gpt2-medium --load_path "path_to_model" --model "DPLoRAGPT2LMHeadModel" --objective "dp_lora"

##### StereoSet
python bias-evaluation/stereoset_debias.py --model_name_or_path "gpt2-medium" --load_path "path_to_model" --model "LoRAGPT2LMHeadModel" --objective "baseline_lora"

python bias-evaluation/experiments/stereoset_evaluation.py --predictions_file "path_to_prediction_file" --output_file "path_to_output_file"

##### SEAT
python bias-evaluation/experiments/seat_debias.py --tests sent-weat6 sent-weat6b sent-weat7 sent-weat7b sent-weat8 sent-weat8b --parametric --model_name_or_path "gpt2-medium" --load_path "path_to_model" --model "LoRAGPT2Model" --objective "cda_lora"





