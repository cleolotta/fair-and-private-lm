# Adapted but modified from Meade et al. (2021) An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models https://github.com/McGill-NLP/bias-bench

from functools import partial
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import transformers
import dp_transformers
from dp_transformers.layers.dp_merged_linear import mark_only_lora_as_trainable
from dp_transformers.module_modification import convert_gpt2_attention_to_lora

 # GPT2Model has different layers than the model I want to load
def prepare_weights(pretrained_state_dict):
    # GPT2Model has different layer names than the models we trained
    # Here, the model weights of the checkpoint we load get manipulated, so that they match those of GPT2Model
    length = len(pretrained_state_dict)
    j = 0
    for key, value in list(pretrained_state_dict.items()):
        if j < length:
            name_old = key
            name_new = name_old.replace('transformer.', '')
            pretrained_state_dict[name_new] = pretrained_state_dict.pop(name_old) 
            j +=1
    pretrained_state_dict.pop('lm_head.weight')
    return(pretrained_state_dict)


class GPT2Model:
    def __new__(self, model_name_or_path):
        return transformers.GPT2Model.from_pretrained(model_name_or_path)

class GPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        return transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
    
class CDAGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model

class DropoutGPT2Model:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        return model

class DropoutGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        return model

class GPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model

class CDAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model
    
class DropoutGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )
        return model
    
class LoRAGPT2Model:
    def __new__(self, model_name_or_path, load_path,lora_dim,lora_alpha, lora_dropout):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        model_path = load_path + "/pytorch_model.bin"
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint = prepare_weights(checkpoint)
        model.load_state_dict(checkpoint, strict=False)
        return model
    
class LoRAptGPT2Model:
    def __new__(self, model_name_or_path,lora_dim,lora_alpha, lora_dropout):
        model = transformers.GPT2Model.from_pretrained(model_name_or_path)
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        return model 
    
class LoRAptGPT2LMHeadModel:
    def __new__(self, model_name_or_path,lora_dim,lora_alpha, lora_dropout):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        return model 
    
class LoRAGPT2LMHeadModel:
    def __new__(self, model_name_or_path, load_path,lora_dim, lora_alpha, lora_dropout):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        model_path = load_path + "/pytorch_model.bin"
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        return model

class LoRAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config, lora_dim, lora_alpha, lora_dropout):
        model = transformers.GPT2ForSequenceClassification.from_pretrained("gpt2-medium", config=config)
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        model_path = model_name_or_path + "/pytorch_model.bin"
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint.pop("lm_head.weight") 
        model.load_state_dict(checkpoint, strict=False)
        return model

    
class DPLoRAGPT2Model:
    def __new__(self, model_name_or_path, load_path, lora_dim, lora_alpha, lora_dropout):
        model = transformers.GPT2Model.from_pretrained(
            model_name_or_path
        )
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        dp_transformers.register_grad_sampler_gpt2_lora()
        model_path = load_path + "/pytorch_model.bin"
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint = prepare_weights(checkpoint)
        model.load_state_dict(checkpoint)
        return model

class DPLoRAGPT2LMHeadModel:
    def __new__(self, model_name_or_path,load_path, lora_dim,lora_alpha, lora_dropout):
        model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        dp_transformers.register_grad_sampler_gpt2_lora()
    
        model_path = load_path + "/pytorch_model.bin"
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint.pop("lm_head.weight") 
        model.load_state_dict(checkpoint, strict=False)
        return model
    
class DPGPT2LMHeadModel:
    def __new__(self, model_name_or_path):
        model = transformers.GPT2LMHeadModel.from_pretrained(
            model_name_or_path
        )
        return model

class DPLoRAGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, lora_dim,lora_alpha, lora_dropout, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained("gpt2-medium", config=config)
        model.state_dict
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        dp_transformers.register_grad_sampler_gpt2_lora()
        model_path = model_name_or_path + "/pytorch_model.bin"
        checkpoint = torch.load(model_path, map_location=device)
        checkpoint.pop("lm_head.weight") 
        model.load_state_dict(checkpoint, strict=False)
        return model

class LoRAptGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, lora_dim,lora_alpha, lora_dropout, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(model_name_or_path, config=config)
        model.state_dict
        model = convert_gpt2_attention_to_lora(
            model, r=lora_dim, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            enable_lora=[True, False, True], merge_weights=False
        )
        mark_only_lora_as_trainable(model)
        return model

class DPGPT2ForSequenceClassification:
    def __new__(self, model_name_or_path, config):
        model = transformers.GPT2ForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
            )
        return model
