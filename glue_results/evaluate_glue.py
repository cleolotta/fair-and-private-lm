import json
import os
import pandas as pd

models = ["pretrained","baseline", "cda","dp", "cda_dp", "dropout", "dropout_dp"]
results = {}
glue = pd.DataFrame(columns={'model', 'test', 'epoch', 'value'})
counter = 0
for model in models:
    for f in os.listdir(f'./glue_results/{model}'):
        dir = f'./glue_results/{model}/' + f
        filename = dir.split("/")
        print(dir.split("/"))
        testname = filename[3].split("_") # 1 = test, 2 = epoch 
        results['model'] = filename[2]
        results['test'] = testname[1]
        results['epoch'] =testname[2]
        f = open(dir)
        data = json.load(f)
        if "cola" in dir:
            print(data["eval_matthews_correlation"])
            results['score'] = "Matthew’s correlation"
            results['value'] = data["eval_matthews_correlation"]
        elif "mrpc" in dir:
            print(data['eval_f1'])
            results['score'] = "F1 Score"
            results['value'] = data['eval_f1']
        elif "stsb" in dir:
            print(data['eval_spearmanr'])
            results['score'] = "Spearman correlation"
            results['value'] = data['eval_spearmanr']
        else:
            print(data["eval_accuracy"])
            results['score'] = "Accuracy"
            results['value'] = data["eval_accuracy"]
        glue = glue.append(results, ignore_index=True)
#glue = glue.set_index('model')
glue_model = glue.groupby(["model","test"]).agg('mean')
glue_model.to_csv('glue_results.txt', decimal=",")