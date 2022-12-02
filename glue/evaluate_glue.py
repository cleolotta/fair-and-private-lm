import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


models = ["_Pre-trained_GPT2","Baseline", "CDA","Dropout","DP", "cda_dp","dropout_dp"]
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
glue_avg = glue_model.groupby("model").agg('mean')
glue_avg = glue_avg.reset_index()
fig = plt.figure(figsize=(10,5))
ax = sns.barplot(data=glue_avg, x="model", y="value", palette='Set2',linewidth=2.5)
ax.set(xlabel="Training Objective",title="GLUE Results", ylabel="GLUE Score ↑")
xlabels = ["Baseline", "CDA", 'DP\n(eps=3.53)', "Dropout","Pre-trained GPT-2", "CDA + DP\n(eps=3.53)","Dropout + \n(eps=3.53)"]
ax.set_xticklabels(xlabels)
plt.show()
glue_model.to_csv('./glue_results/glue_results.txt', decimal=".")