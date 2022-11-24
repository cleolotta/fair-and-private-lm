import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



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
data_cda = get_data_mia("./models/cda_lora_model" + "/stdout",hue='CDA')
data_dropout = get_data_mia("./models/dropout_lora_model" + "/stdout", hue = 'Dropout')
data_dp = get_data_mia("./models/dp_model" + "/stdout", hue = 'DP')
data_baseline =  get_data_mia("./models/baseline_lora_model" + "/stdout", hue = 'Baseline')
data_cda_dp = get_data_mia("./models/alt_cda_dp_model" + "/stdout", hue = 'CDA + DP')
data_dropout_dp = get_data_mia("./models/dropout_dp_model" + "/stdout", hue = 'Dropout + DP')

df_baseline = pd.DataFrame.from_dict(data_baseline)
df_cda_dp = pd.DataFrame.from_dict(data_cda_dp)
df_dropout_dp = pd.DataFrame.from_dict(data_dropout_dp)
df_cda = pd.DataFrame.from_dict(data_cda)
df_dp = pd.DataFrame.from_dict(data_dp)
df_dropout = pd.DataFrame.from_dict(data_dropout)



# RQ 1 Does training with differential privacy objective lead to more fairness in a model?
rq1 = pd.read_excel("C:/Users/cmatz/results.xlsx", sheet_name="RQ1")
rq1 = rq1.set_index("eval_method")
rq1 = rq1.T.reset_index()
plt.figure(figsize=(10,5))
ax = sns.barplot(data=rq1, x="index", y="SEAT average effect size",palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="SEAT average effect size", ylabel="SEAT average ↓")
plt.show()

plt.figure(figsize=(10,5))
ax = sns.lineplot(data=rq1, x="index", y="Stereoset Score", palette='Set2',linewidth=2.5)
#ax.axhline(y=50, color= 'b', linewidth=1, )
ax = sns.lineplot(data=rq1, x="index", y="BEC-Pro", palette='Set2',linewidth=2.5)

ax.set(xlabel="Model",title="StereoSet/BEC-Pro Score", ylabel="StereoSet/BEC-Pro Score")
ax.legend(title='Evaluation Metric',labels=['StereoSet Score','BEC-Pro Score'])

plt.show()

#RQ2
ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall  vs. Epoch'
var = 'attack1'
plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=df_baseline[var],  palette='Set2',linewidth=2.5)
ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_baseline[var]))])
ax2 = sns.lineplot(data=df_cda[var], palette='Set2', linewidth=2.5)
ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_cda[var]))])
ax3 = sns.lineplot(data=df_dropout[var], palette='Set2', linewidth=2.5)
ax3.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dropout[var]))])
ax4 = sns.lineplot(data=df_dp[var], palette='Set2', linewidth=2.5)
ax4.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dp[var]))])
ax4.legend(title='Model / Training objective',labels=['Baseline','CDA', 'Dropout', 'DP'])
plt.show()


#RQ3.1
rq3 = pd.read_excel("C:/Users/cmatz/results.xlsx", sheet_name="RQ3.1")
rq3 = rq3.set_index("eval_method")
rq3 = rq3.T.reset_index()

plt.figure(figsize=(10,5))
ax = sns.barplot(data=rq3, x="index", y="SEAT average", palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="SEAT average effect size", ylabel="SEAT average ↓")
plt.show()

plt.figure(figsize=(10,5))
ax = sns.lineplot(data=rq3, x="index", y="Stereoset_score", palette='Set2',linewidth=2.5)
#ax.axhline(y=50, color= 'b', linewidth=1, )
ax = sns.lineplot(data=rq3, x="index", y="BEC-Pro", palette='Set2',linewidth=2.5)

ax.set(xlabel="Model",title="StereoSet/BEC-Pro Score", ylabel="StereoSet/BEC-Pro Score")
ax.legend(title='Evaluation Metric',labels=['StereoSet Score','BEC-Pro Score'])

plt.show()

#RQ3.2

ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall  vs. Epoch'
var = 'attack1'
plt.figure(figsize=(10, 5))
ax = sns.lineplot(data=df_baseline[var],  palette='Set2',linewidth=2.5)
ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_baseline[var]))])
ax2 = sns.lineplot(data=data_cda_dp[var], palette='Set2', linewidth=2.5)
ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_cda_dp[var]))])
ax3 = sns.lineplot(data=data_dropout_dp[var], palette='Set2', linewidth=2.5)
ax3.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_dropout_dp[var]))])
ax4 = sns.lineplot(data=df_dp[var], palette='Set2', linewidth=2.5)
ax4.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(df_dp[var]))])
ax4.legend(title='Model / Fine-tuning Method',labels=['Baseline','CDA + DP', 'Dropout + DP', 'DP'])
plt.show()

#RQ4
rq4 = pd.read_excel("C:/Users/cmatz/results.xlsx", sheet_name="RQ4")
rq4 = rq4.set_index("eval_method")
rq4 = rq4.T.reset_index()

plt.figure(figsize=(10,5))
ax = sns.barplot(data=rq4, x="index", y="Perplexity", palette='Set2',linewidth=2.5)
ax.set(xlabel="Model",title="Perplexity", ylabel="Perplexity ↓")
plt.show()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_xlim(0, 8)
dim = np.arange(1,8)
lns1 = ax.plot(dim,rq4['LM Score'], color="royalblue", label="LM Score",linewidth=2.5)
ax2 = ax.twinx()
lns2 = ax2.plot(dim,rq4['ICAT Score'], color="coral", label="ICAT Score",linewidth=2.5)
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
xlabels = [x for x in rq4['index'].tolist()]
ax.set_xticklabels([" ", 'Pre-trained GPT-2 ', 'Baseline', 'DP', 'CDA', 'CDA + DP', 'Dropout', 'Dropout + DP'])
ax.set_ylabel("LM Score ↑")
ax2.set_ylabel("ICAT Score ↑")
ax.set_title("LM / ICAT Score")
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot(data=rq4, x="SEAT average effect size", hue="index",y="LM Score", palette='Set2', s=150)
ax.set(xlabel="SEAT ↓",title="SEAT vs. LM score", ylabel="LM score ↑")
ax.legend(title="Training objective")

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot(data=rq4, x="StereoSet Score", hue="index",y="LM Score", palette='Set2', s=150)
ax.set(xlabel="Stereoset - least bias at 50%",title="Stereoset vs. LM score", ylabel="LM score ↑")
ax.legend(title="Training objective")

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot(data=rq4, x="BEC-Pro Score", hue="index",y="LM Score", palette='Set2', s=150)
ax.set(xlabel="Stereoset - least bias at 50%",title="Stereoset vs. LM score", ylabel="LM score ↑")
ax.legend(title="Training objective")
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot(data=rq4, x="LM Score", hue="index",y="EoT MIA Recall", palette='Set2', s=150)
ax.set(xlabel="LM score ↑",title="Leakage vs. LM score", ylabel="Leakage ↓")
ax.legend(title="Training objective")
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot(data=rq4, x="Perplexity", hue="index",y="EoT MIA Recall", palette='Set2', s=150)
ax.set(xlabel="Perplexity ↓",title="Leakage vs. Perplexity", ylabel="Leakage ↓")
ax.legend(title="Training objective")
plt.show()

# Main findings:
ind_list = [2, 4,6]
rq4_ = rq4.iloc[ind_list]
fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot( data=rq4[1:], x="BEC-Pro Score", hue="index",y="EoT MIA Recall", palette='Set2', s=150)
ax.set(xlabel="BEC-Pro Score",title="Leakage vs. BEC-Pro Score", ylabel="Leakage ↓")
ax.legend(title="Training objective")
ax.set_ylim(0.04,0.08)
ax.set_xlim(40,60)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot( data=rq4[1:], x="SEAT average effect size", hue="index",y="EoT MIA Recall", palette='Set2', s=150)
ax.set(xlabel="SEAT average effect size ↓",title="Leakage vs. SEAT average effect size", ylabel="Leakage ↓")
ax.legend(title="Training objective")
ax.set_ylim(0.04,0.08)
ax.set_xlim(0,0.2)
plt.show()

fig = plt.figure(figsize=(10,5))
ax = sns.scatterplot( data=rq4[1:], x="StereoSet Score", hue="index",y="EoT MIA Recall", palette='Set2', s=150)
ax.set(xlabel="StereoSet Score",title="Leakage vs. StereoSet Score", ylabel="Leakage ↓")
ax.legend(title="Training objective")
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
ax4.legend(title='Model / Fine-tuning Method',labels=['Baseline','CDA', 'CDA + DP', 'DP'])
plt.show()