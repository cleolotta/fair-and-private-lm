import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FormatStrFormatter



def get_data_mia(filename, hue): 
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

data_cda = get_data_mia("C:/Users/cmatz/master-thesis/models/cda_lora_model" + "/stdout",hue='CDA')
data_dropout = get_data_mia("C:/Users/cmatz/master-thesis//models/dropout_lora_model" + "/stdout", hue = 'Dropout')
data_dp = get_data_mia("C:/Users/cmatz/master-thesis//models/dp_model" + "/stdout", hue = 'DP')
data_baseline =  get_data_mia("C:/Users/cmatz/master-thesis//models/baseline_lora_model" + "/stdout", hue = 'Baseline')
data_cda_dp = get_data_mia("C:/Users/cmatz/master-thesis//models/cda_dp_model" + "/stdout", hue = 'CDA + DP')
data_dropout_dp = get_data_mia("C:/Users/cmatz/master-thesis//models/dropout_dp_model" + "/stdout", hue = 'Dropout + DP')

df_baseline = pd.DataFrame.from_dict(data_baseline)
df_cda_dp = pd.DataFrame.from_dict(data_cda_dp)
df_dropout_dp = pd.DataFrame.from_dict(data_dropout_dp)
df_cda = pd.DataFrame.from_dict(data_cda)
df_dp = pd.DataFrame.from_dict(data_dp)
df_dropout = pd.DataFrame.from_dict(data_dropout)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
sns.set(rc={"font.size":18, "font.family": "DejaVu Sans", 
            "axes.titlesize":18,"axes.labelsize":15, "ytick.labelsize":15, 
            "xtick.labelsize":15 , 'legend.fontsize':15, 'legend.title_fontsize': 15}, style="white")


df = pd.read_csv("./bias-evaluation/results/all_results.csv", sep=";")

# RQ 1 Does training with differential privacy objective lead to more fairness in a model?
l = [3,5,6,7,8,9,10,11,12]
rq1 = df.iloc[l]
rq1 = rq1.set_index("eval_method")
rq1 = rq1.T.reset_index()
plt.figure(figsize=(12,7))
ax = sns.barplot(data=rq1[:5], x="index", y="SEAT average effect size",palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="SEAT average effect size", ylabel="SEAT average ↓")
plt.show()

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq1[:5], x="SEAT average effect size", hue="index",y="BEC-Pro Score", palette='Set2', s=150)
ax.set(xlabel="SEAT average effect size",title="SEAT average effect size vs. BEC-Pro score", ylabel="BEC-Pro score")
ax.legend(title="Training objective")
plt.show()

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq1[:5], x="Stereoset Score", hue="index",y="BEC-Pro Score", palette='Set2', s=150)
ax.set(xlabel="StereoSet score",title="StereoSet vs. BEC-Pro score", ylabel="BEC-Pro score")
ax.legend(title="Training objective")
plt.show()

#RQ2
ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall  vs. Epoch'
var = 'attack1'
plt.figure(figsize=(12,7))

ax = sns.lineplot(data=df_baseline[var],  palette='Set2',linewidth=2.5)
ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_baseline[var]))])
ax2 = sns.lineplot(data=df_cda[var], palette='Set2', linewidth=2.5)
ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_cda[var]))])
ax3 = sns.lineplot(data=df_dropout[var], palette='Set2', linewidth=2.5)
ax3.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dropout[var]))])
ax4 = sns.lineplot(data=df_dp[var], palette='Set2', linewidth=2.5)
ax4.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dp[var]))])
ax4.legend(title='Model / Training objective',labels=['Baseline','CDA', 'Dropout', 'DP'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()


#RQ3.1
rq3 = df.iloc[l]
rq3 = rq3.set_index("eval_method")
rq3 = rq3.T.reset_index()
l2 = [0,1,2,3,5,6]

plt.figure(figsize=(15,8))
ax = sns.barplot(data=rq3.iloc[l2], x="index", y="SEAT average effect size", palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="SEAT average effect size", ylabel="SEAT average ↓")
ax.set_xticklabels(['Pre-trained \nGPT-2 ', 'Baseline', 'CDA', 'Dropout', 'CDA + DP\n(eps=3.53)', 'Dropout + DP\n(eps=3.53)'])

plt.show()

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq3.iloc[l2], x="Stereoset Score", hue="index",y="BEC-Pro Score", palette='Set2', s=150)
ax.set(xlabel="StereoSet Score",title="StereoSet vs. BEC-Pro score", ylabel="BEC-Pro score")
ax.legend(title="Training objective")

plt.show()

#RQ3.2

ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall  vs. Epoch'
var = 'attack1'
fig = plt.figure(figsize=(12,7))
ax = sns.lineplot(data=df_baseline[var],  palette='Set2',linewidth=2.5)
ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_baseline[var]))])
ax2 = sns.lineplot(data=data_cda_dp[var], palette='Set2', linewidth=2.5)
ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_cda_dp[var]))])
ax3 = sns.lineplot(data=data_dropout_dp[var], palette='Set2', linewidth=2.5)
ax3.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_dropout_dp[var]))])
ax4 = sns.lineplot(data=df_dp[var], palette='Set2', linewidth=2.5)
ax4.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dp[var]))])
ax4.legend(loc=1,title='Model / Fine-tuning Method',labels=['Baseline','CDA + DP (eps=3.53)', 'Dropout + DP (eps=3.53)', 'DP (eps=3.53)'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()

#RQ4
l3 = [0, 1,2,4]
rq4 = df.iloc[l3].set_index("eval_method")
rq4 = rq4.T.reset_index()

plt.figure(figsize=(12, 7))
ax = sns.barplot(data=rq4[1:], x="index", y="EoT Perplexity", palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="Perplexity", ylabel="Perplexity ↓")
ax.set_xticklabels(['Baseline', 'CDA', 'Dropout', 'DP', 'CDA + DP', 'Dropout + DP'])

plt.show()

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4, x="EoT Perplexity", hue="index",y="ICAT", palette='Set2', s=150)
ax.set(xlabel="Perplexity ↓",title="Perplexity vs. ICAT", ylabel="ICAT ↑")
ax.legend(title="Training objective")
plt.show()

fig = plt.figure(figsize=(14,12))
ax = fig.add_subplot(111)
#ax.set_xlim(0, 8)
#dim = np.arange(1,8)
lns1 = ax.plot(rq4['LM score'], color="royalblue", label="LM Score ↑",linewidth=2.5)
ax2 = ax.twinx()
lns2 = ax2.plot(rq4['ICAT'], color="coral", label="ICAT Score ↑",linewidth=2.5)
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
xlabels = [x for x in rq4['index'].tolist()]
ax.set_xticklabels([" ", 'Pre-trained \nGPT-2 ', 'Baseline', 'CDA', 'Dropout', 'DP', 'CDA+DP', 'Dropout+DP'])

ax.set_ylabel("LM Score ↑")
ax.set_xlabel("Model")
ax2.set_ylabel("ICAT Score ↑")
ax.set_title("LM / ICAT Score")
plt.show()

l3 = [0,1,2,3,4,5,12]
rq4 = df.iloc[l3].set_index("eval_method").T.reset_index()
fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4, x="SEAT average effect size", hue="index",y="LM score", palette='Set2', s=150)
ax.set(xlabel="SEAT ↓",title="SEAT vs. LM score", ylabel="LM score ↑")
ax.legend(title="Training objective")
plt.show()

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4, x="Stereoset Score", hue="index",y="LM score", palette='Set2', s=150)
ax.set(xlabel="Stereoset - least bias at 50%",title="Stereoset vs. LM score", ylabel="LM score ↑")
ax.legend(title="Training objective")

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4, x="BEC-Pro Score", hue="index",y="LM score", palette='Set2', s=150)
ax.set(xlabel="BEC-Pro - least bias at 50%",title="BEC-Pro vs. LM score", ylabel="LM score ↑")
ax.legend(title="Training objective")

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4[1:], x="BEC-Pro Score", hue="index",y="EoT Perplexity", palette='Set2', s=150)
ax.set(xlabel="BEC-Pro Score - least bias at 50%",title="BEC-Pro Score vs. Perplexity", ylabel="Perplexity ↓")
ax.legend(title="Training objective")

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4, x="LM score", hue="index",y="EoT MIA recall", palette='Set2', s=150)
ax.set(xlabel="LM score ↑",title="Leakage vs. LM score", ylabel="Leakage ↓")
ax.legend(title="Training objective")

fig = plt.figure(figsize=(12,7))
ax = sns.scatterplot(data=rq4[1:], x="EoT MIA recall", hue="index",y="EoT Perplexity", palette='Set2', s=150)
ax.set(xlabel="Leakage ↓",title="Leakage vs. Perplexity", ylabel="Perplexity ↓")
ax.legend(title="Training objective")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()

# RQ5
glue = pd.read_csv("./glue_results/glue_results.txt")
glue_avg = glue.groupby("model").agg('mean')
glue_avg = glue_avg.reset_index()
fig = plt.figure(figsize=(12,7))
ax = sns.barplot(data=glue_avg, x="model", y="value", palette='Set2',linewidth=2.5)
ax.set(xlabel="Training Objective",title="GLUE Results", ylabel="GLUE Score ↑")
xlabels = ["Baseline", "CDA", 'DP', "Dropout","Pre-trained \nGPT-2", "CDA + DP","Dropout + DP"]
ax.set_xticklabels(xlabels)
plt.show()



# Main findings:
ind_list = [2, 4,6]
rq4_ = rq4.iloc[ind_list]
fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot( data=rq4[1:], x="BEC-Pro Score", hue="index",y="EoT MIA recall", palette='Set2', s=150)
ax.set(xlabel="BEC-Pro Score",title="Leakage vs. BEC-Pro Score", ylabel="Leakage ↓")
ax.legend(loc=1, title="Training objective")
ax.set_ylim(0.04,0.08)
ax.set_xlim(40,60)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot( data=rq4[1:], x="SEAT average effect size", hue="index",y="EoT MIA recall", palette='Set2', s=150)
ax.set(xlabel="SEAT average effect size ↓",title="Leakage vs. SEAT average effect size", ylabel="Leakage ↓")
ax.legend(loc=1, title="Training objective")
ax.set_ylim(0.04,0.08)
ax.set_xlim(0,0.2)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot( data=rq4[1:], x="Stereoset Score", hue="index",y="EoT MIA recall", palette='Set2', s=150)
ax.set(xlabel="StereoSet Score",title="Leakage vs. StereoSet Score", ylabel="Leakage ↓")
ax.legend(loc=1,title="Training objective")
ax.set_ylim(0.04,0.08)
ax.set_xlim(60,70)
plt.show()

ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall  vs. Epoch'
var = 'attack1'
plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=df_baseline[var],  palette='Set2',linewidth=2.5)
ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_baseline[var]))])
ax2 = sns.lineplot(data=data_cda[var], palette='Set2', linewidth=2.5)
ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_cda[var]))])
ax3 = sns.lineplot(data=data_cda_dp[var], palette='Set2', linewidth=2.5)
ax3.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_cda_dp[var]))])
ax4 = sns.lineplot(data=df_dp[var], palette='Set2', linewidth=2.5)
ax4.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dp[var]))])
ax4.legend(loc=1,title='Model / Fine-tuning Method',labels=['Baseline','CDA', 'CDA + DP', 'DP'])
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

plt.show()