from socket import AF_UNIX
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("C:/Users/cmatz/results.csv", header=0, sep=";")
df = df.set_index("eval_method")
df_seat = df[5:12]

df_lm = df[:2]
df_mia = df[2:3]
df_bias1 = df[3:4]
df_bias2 = df[4:5]
df_bias1_2 = df[3:5]
df_glue = df[-1:]
#mean = list(round(df_seat.abs().mean(axis=0),4))
#new_row = {}
#new_row['eval_method'] = "SEAT_avg"
#for col, item in zip(df_seat.columns, mean):
#    new_row[col] = item
#df_seat = df_seat.reset_index()
#df_seat = df_seat.append(new_row,ignore_index=True)   
#df_seat = df_seat.set_index("eval_method")


ax = df_bias1_2.plot.bar(rot=0, title="Stereoset vs. BEC-Pro Score")
ax.axhline(y=50, color= 'black', linewidth=2.5, )
ax.text(x= 0.4, y=53, s='no bias at 50%')
plt.show()

ax = df_seat.plot.bar(rot=0, title="SEAT scores")
ax.axhline(y=0, color= 'black', linewidth=1)
ax.text(x= 0.25, y=-0.08, s='the closer to 0, \n the less bias')
plt.show()

df_avg = df_seat[-1:]
ax = df_avg.transpose().plot(rot=0, linewidth=2.5, title="SEAT average")
#ax.axhline(y=0, color= 'black', linewidth=1, )
#ax.text(x= 0.25, y=-0.08, s='the closer to 0, \n the less bias')
plt.show()

df_seat_names = df_seat.transpose()[["SEAT6", "SEAT7", "SEAT8"]].transpose()
df_seat_terms = df_seat.transpose()[["SEAT6b", "SEAT7b", "SEAT8b"]].transpose()

round(df_seat_names.abs().mean(),4).plot(linewidth=2.5, title="SEAT average names")
plt.show()
round(df_seat_terms.abs().mean(),4).plot(linewidth=2.5, title="SEAT average terms")
plt.show()

df = df.reset_index()
df_mia_bias = df[(df['eval_method']=='GLUE') | (df['eval_method']=="SEAT average")].transpose()


df = df.T.reset_index()


plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.lineplot(data=df, x="index", y="SEAT average", palette='Set2')
ax.set(xlabel="Model",title="SEAT average", ylabel="SEAT average ↓")
plt.show()

ax = df_seat.plot.bar(rot=0, title="SEAT scores")
ax.axhline(y=0, color= 'black', linewidth=1)
ax.text(x= 0.25, y=-0.08, s='the closer to 0, \n the less bias')
plt.show()



# RQ 1 Does training with differential privacy objective lead to more fairness in a model?
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.lineplot(data=df[:3], x="index", y="SEAT average", palette='Set2')
ax.set(xlabel="Model",title="RQ1: Does training with differential privacy objective lead to more fairness in the resulting model?", ylabel="SEAT average ↓")
plt.show()
ax = sns.lineplot(data=df[:3], x="index", y="Stereoset_score", palette='Set2')
ax.axhline(y=50, color= 'b', linewidth=1, )
ax.set(xlabel="Model",title="RQ1: Does training with differential privacy objective lead to more fairness in the resulting model?", ylabel="Stereoset Score - least bias at 50%")
plt.show()

# RQ 2 Does training with debiasing objective lead to less leakage in the model?
l = [1,3,5]
df_ = df.iloc[l]
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.lineplot(data=df_, x="index", y="eot_mia_recall", palette='Set2')
ax.set(xlabel="Model",title="RQ2: Does training with debiasing objective lead to less leakage in the resulting model?", ylabel="Leakage ↓")
plt.show()

#RQ 3.1 How does training with debiasing as well as dp objective affect fairness in a model?
l = [1,4,6]
df_ = df.iloc[l]
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df_, x="eot_mia_recall", hue="index",y="Stereoset_score", palette='Set1', s=150)
ax.set(xlabel="Leakage",title="RQ3: How does training with debiasing as well as dp objective affect fairness in the resulting model?", ylabel="Stereoset Score - least bias at 50%")
#ax.axhline(y=50, color= 'black', linewidth=2.5, )
ax.legend(title="Model")
plt.show()

df_ = df.iloc[l]
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df_, x="eot_mia_recall", hue="index",y="SEAT average", palette='Set1', s=150)
ax.set(xlabel="Leakage",title="RQ3: How does training with debiasing as well as dp objective affect fairness in the resulting model?", ylabel="SEAT average ↓")
ax.legend(title="Model")
plt.show()

#RQ 3.2 How does training with debiasing as well as dp objective affect privacy in a model?
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.barplot(data=df_, x="index", y="eot_mia_recall", palette='Set2')
ax.set(xlabel="Model",title="RQ3: How does training with debiasing as well as dp objective affect privacy in the resulting model?", ylabel="MIA Recall ↓", fontsize=100)
plt.show()

# RQ 4
l = [0,1,2,3,5]
df_ = df.iloc[l]
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.lineplot(data=df, x="index", y="LM score", palette='Set2')
ax.set(xlabel="Model",title="RQ4: LM score (Meade et. al)", ylabel="LM score ↑", fontsize=100)
plt.show()

plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.lineplot(data=df[1:], x="index", y="eot_perplexity", palette='Set2')
ax.set(xlabel="Model",title="RQ4: Perplexity (Mireshghallah et. al)", ylabel="Perplexity ↓")
plt.show()

plt.figure(figsize=(6, 4), tight_layout=True)
ax = sns.lineplot(data=df, x="index", y="LM score", palette='Set2')
ax.set(xlabel="Model",title="Perplexity (Mireshghallah et. al)", ylabel="Perplexity")
plt.show()

fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
lns1 = sns.lineplot(data=df, x="index", y="eot_perplexity", ax=ax, label="Perplexity")
ax2 = ax.twinx()
lns2 = sns.lineplot(data=df, x="index", y="LM score", ax=ax2, color="r", label="LM score")
ax.set(xlabel="Model",title="Language Modelling Ability", ylabel="LM score ↑ (Meade et. al)")
ax2.set(xlabel="Model", ylabel="Perplexity ↓ (Mireshghallah et. al)")
plt.show()

# RQ 5
plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.lineplot(data=df, x="index", y="GLUE", palette='Set2')
ax.set(xlabel="Model",title="RQ5: How does training with debiasing and/or dp objective affect the resulting model‘s ability to perform downstream NLU tasks?", ylabel="GLUE average ↓")
plt.show()

# Metrics vs. GLUE

plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df, x="SEAT average", hue="index",y="GLUE", palette='Set2', s=150)
ax.set(xlabel="SEAT ↓",title="SEAT vs. GLUE", ylabel="GLUE ↑")
ax.legend(title="Model")

plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df, x="Stereoset_score", hue="index",y="GLUE", palette='Set2', s=150)
ax.set(xlabel="Stereoset - least bias at 50%",title="Stereoset vs. GLUE", ylabel="GLUE ↑")
ax.legend(title="Model")

plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df, x="eot_mia_recall", hue="index",y="GLUE", palette='Set2', s=150)
ax.set(xlabel="Leakage ↓",title="Leakage vs. GLUE", ylabel="GLUE ↑")
ax.legend(title="Model")
plt.show()

# LM ability vs. other metrics

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=df, x="SEAT average", hue="index",y="LM score", palette='Set2', s=150)
ax.set(xlabel="SEAT ↓",title="SEAT vs. LM score", ylabel="LM score ↑")
ax.legend(title="Model")

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=df, x="Stereoset_score", hue="index",y="LM score", palette='Set2', s=150)
ax.set(xlabel="Stereoset - least bias at 50%",title="Stereoset vs. LM score", ylabel="LM score ↑")
ax.legend(title="Model")

plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df, x="LM score", hue="index",y="eot_mia_recall", palette='Set2', s=150)
ax.set(xlabel="LM score ↑",title="Leakage vs. LM score", ylabel="Leakage ↓")
ax.legend(title="Model")
plt.show()

plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.scatterplot(data=df, x="eot_perplexity", hue="index",y="eot_mia_recall", palette='Set2', s=150)
ax.set(xlabel="Perplexity ↓",title="Leakage vs. Perplexity", ylabel="Leakage ↓")
ax.legend(title="Model")
plt.show()

# Privacy vs. Bias

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=df, x="eot_mia_recall", hue="index",y="Stereoset_score", palette='Set2', s=150)
ax.set(xlabel="Leakage ↓",title="Leakage vs. Bias 1", ylabel="Stereoset - least bias at 50%")
ax.legend(title="Model")

plt.figure(figsize=(10,6), tight_layout=True)
ax = sns.scatterplot(data=df, x="eot_mia_recall", hue="index",y="SEAT average", palette='Set2', s=150)
ax.set(xlabel="Leakage ↓",title="Leakage vs. Bias 2", ylabel="SEAT average ↓")
ax.legend(title="Model")
plt.show()


plt.figure(figsize=(6,4), tight_layout=True)
ax = sns.barplot(data=rq4, x="index", y="eot_mia_recall", palette='Set2')
ax.set(xlabel="Model",title="RQ3: How does training with debiasing as well as dp objective affect privacy in the resulting model?", ylabel="MIA Recall ↓", fontsize=100)
plt.show()
