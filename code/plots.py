import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter



def get_data_mia(filename, hue): 
    # Source: https://github.com/mireshghallah/ft-memorization
    data_mia = {'attack1': [], 'attack2': [], 'valperp': [], 'trainperp': [], 'recall1': [], 'recall2': [], 'precision1': [], 'precision2': [], 'gen_gap':[], 'hue':[]}
    with open(filename, 'r') as file:
        count = 0
        last_pos = file.tell()
        f = file.readline()
        while f != '':
            file.seek(last_pos)
            data_point = file.readline().strip()
            if data_point == '*************end of training':
                break
            if data_point == '____':
                count += 1
                data = []
                for i in range(8):
                    data_point = file.readline().strip()
                    if (data_point  != '_____') and data_point != '':
                        data.append(float(data_point))
                data_mia['attack1'].append(data[0])
                data_mia['attack2'].append(data[1])
                data_mia['valperp'].append(data[2])
                data_mia['trainperp'].append(data[3])
                data_mia['recall1'].append(data[4])
                data_mia['recall2'].append(data[5])
                data_mia['precision1'].append(data[6])
                data_mia['precision2'].append(data[7])
                data_mia['gen_gap'].append(data[2]-data[3])
                data_mia['hue'].append(hue)            
            last_pos = file.tell()
            f = file.readline()
    return data_mia

data_cda = get_data_mia("./models/cda_2_sided_lora" + "/stdout",hue='CDA')
data_dropout = get_data_mia("./models/dropout_lora_model" + "/stdout", hue = 'Dropout')
data_dp = get_data_mia("./models/dp_model" + "/stdout", hue = 'DP')
data_baseline =  get_data_mia(".//models/baseline_lora_model" + "/stdout", hue = 'Baseline')
data_cda_dp = get_data_mia("./models/cda_2-sided_dp_model" + "/stdout", hue = 'CDA + DP')
data_dropout_dp = get_data_mia("./models/dropout_dp_model" + "/stdout", hue = 'Dropout + DP')

df_baseline = pd.DataFrame.from_dict(data_baseline)
df_cda_dp = pd.DataFrame.from_dict(data_cda_dp)
df_dropout_dp = pd.DataFrame.from_dict(data_dropout_dp)
df_cda = pd.DataFrame.from_dict(data_cda)
df_dp = pd.DataFrame.from_dict(data_dp)
df_dropout = pd.DataFrame.from_dict(data_dropout)


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set(rc={"font.size":30, 
            "axes.titlesize":30,"axes.labelsize":25, "ytick.labelsize":20, 
            "xtick.labelsize":20 , 'legend.fontsize':20, 'legend.title_fontsize': 20}, style="white")

# read data 
df = pd.read_csv("./bias-evaluation/results/all_results.csv", sep=";")

ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall w.r.t. Epoch'
var = 'attack1'
plt.figure(figsize=(12,7))
ax = sns.lineplot(data=df_baseline[var],  palette="Set2",linewidth=2.5)
ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_baseline[var]))])
ax2 = sns.lineplot(data=df_cda[var], palette='Set2', linewidth=2.5)
ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_cda[var]))])
ax3 = sns.lineplot(data=df_dropout[var], palette='Set2', linewidth=2.5)
ax3.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dropout[var]))])
ax4 = sns.lineplot(data=df_dp[var], palette='Set2', linewidth=2.5)
ax4.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dp[var]))])
ax5 = sns.lineplot(data=data_cda_dp[var], palette='Set2', linewidth=2.5)
ax5.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_cda_dp[var]))])
ax6 = sns.lineplot(data=data_dropout_dp[var], palette='Set2', linewidth=2.5)
ax6.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_dropout_dp[var]))])
ax6.legend(title='Model',labels=['Baseline','CDA', 'Dropout', 'DP', 'CDA + DP', 'Dropout + DP'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.grid()
#plt.show()
plt.savefig("mia_recall.pdf", transparent=True)

l3 = [0,1,2,3,4,5,12]
rq4 = df.iloc[l3].set_index("eval_method").T.reset_index()

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4[1:7], x="BEC-Pro Score", hue="index",y="EoT Perplexity", palette='Set2', s=300)
ax.set(xlabel="BEC-Pro Score - least bias at 50%",title="BEC-Pro Score w.r.t. Perplexity", ylabel="Perplexity ↓")
ax.legend(title="Model",markerscale=2., scatterpoints=1)
#plt.show()
plt.grid()
plt.savefig("bec-perplexity.pdf", transparent=True)

fig = plt.figure(figsize=(12, 7))
ax = sns.scatterplot(data=rq4[1:7], x="EoT MIA recall", hue="index",y="EoT Perplexity", palette='Set2', s=300)
ax.set(xlabel="Leakage ↓",title="Leakage w.r.t. Perplexity", ylabel="Perplexity ↓")
ax.legend(title="Model",markerscale=2., scatterpoints=1)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
plt.grid()
plt.savefig("leakage-perplexity.pdf", transparent=True)


# RQ5
glue = pd.read_csv("./glue/glue_results.txt")
glue_avg = glue.groupby("model").agg('mean').round(decimals=2)
glue_avg = glue_avg.reset_index()
fig = plt.figure(figsize=(12,7))
ax = sns.barplot(data=glue_avg, x="model", y="value", palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="GLUE Results", ylabel="GLUE Score ↑")
xlabels = ["Baseline", "CDA", 'DP', "Dropout","pt.GPT-2", "CDA+DP","Dropout+DP"]
for i in ax.containers:
    ax.bar_label(i,)
ax.set_xticklabels(xlabels)
plt.show()



