import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
# Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
# Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
# Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
# Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY


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
    
    
def get_data_exposure(filename,hue,cnt_s = 20,old=False):
    data_exposure = {'exposure': [] ,'attack1': [], 'valperp': [], 'trainperp': [], 'hue':[]}
    cnt = 0
    with open(filename, 'r') as file:
        last_pos = file.tell()
        f = file.readline()
        while f != '':
            if cnt>cnt_s:
                    break
            file.seek(last_pos)
            data_point = file.readline().strip()
            
            if data_point == 'running canary eval':
                data = []
                data_exposure['exposure'].append(float(file.readline().strip()))
            
            if data_point == '____':
                data = []
                le = 4 if old else 3
                for i in range(le):
                    data_point = file.readline().strip()
                    data.append(float(data_point))
                
                data_exposure['attack1'].append(data[0])
                
                if not old:
                    data_exposure['valperp'].append(data[1])
                    data_exposure['trainperp'].append(data[2])
                else:
                    data_exposure['valperp'].append(data[2])
                    data_exposure['trainperp'].append(data[3])                   
                data_exposure['hue'].append(hue)
                cnt+=1
 
                
            last_pos = file.tell()
            f = file.readline()
            
    return data_exposure


def plot_linear(xlabel,ylabel,title,var,data_mia_head,data_mia_full,f_name):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=data_mia_head[var],  palette='Set2',linewidth=2.5)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_mia_head[var]))])
    ax2 = sns.lineplot(data=data_mia_full[var], palette='Set2', linewidth=2.5)
    ax2.set(xlabel=xlabel, ylabel=ylabel, title=title, xticks=[i+1 for i in range(len(data_mia_full[var]))])
    ax2.legend(title='Fine-tuning Method',labels=['FT with augmented data','FT with not-augmented data'])
    plt.savefig(f_name, transparent=True)

    plt.show()

def plot_scatter(xlabel,ylabel,title,hue,legendtitle,df_all,f_name,pareto=False, xy=(0,0), xytext=(0,0)):
    
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    sns.set(rc={"font.size":18, "font.family": "serif", 
                "axes.titlesize":18,"axes.labelsize":18, "ytick.labelsize":18, 
                 "xtick.labelsize":18 , 'legend.fontsize':18, 'legend.title_fontsize': 18}, style="white")
    
    p_front = pareto_frontier(list(df_all[xlabel]), list(df_all[ylabel]), maxX = False, maxY = False) 

    plt.figure(figsize=(10,6), tight_layout=True)
    ax = sns.scatterplot(data=df_all, x=xlabel, y=ylabel,hue=hue, palette='Set2', s=60)
    ax.set(xlabel=xlabel,title=title, ylabel=ylabel)
    ax.legend(title=legendtitle)#loc='lower right' 
    if pareto:
        plt.plot(p_front[0], p_front[1], zorder=2, linewidth=8, color='c', alpha=0.2)
        # plt.text(p_front[0][0], p_front[1][0], "Pareto Frontier", horizontalalignment='left', size='medium', color='b')
        ax.annotate("Pareto Frontier",
            xy=xy, xycoords='data',
            xytext=xytext, textcoords='data',
            color='black',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3", color='black'),
            )
    plt.savefig(f_name, transparent=True)

    plt.show()
    
    
data_mia_full_augmented = get_data_mia("C:/Users/cmatz/master-thesis/fair-and-private-lm/debiasing/ft_gpt2_mia_aug_trained/canary_None_5_adapter_False_head_False_layer_None_ref_True_maxlen_1024_red_None_model_gpt2_lr_1e-05_epoch_5_trba_1_acc_8_evba1_data_None/stdout",hue='FT with augmented data')
data_mia_full_original =  get_data_mia("C:/Users/cmatz/master-thesis/fair-and-private-lm/debiasing/ft_gpt2-not_augmented-3epochs/stdout", hue = 'FT with not augmented data')


df_augmented = pd.DataFrame.from_dict(data_mia_full_augmented)
df_original = pd.DataFrame.from_dict(data_mia_full_original)


df_all= pd.concat([df_augmented, df_original], axis=0)
#data_exposure = get_data_exposure('/home/tr33/Documents/efficient_ft/gen/stdout-exposure')    


ylabel = 'MIA Recall'
xlabel='Epoch'
title = 'MIA recall  vs. Epoch'
var = 'attack1'
f_name ='mia-epoch.pdf'
plot_linear(xlabel,ylabel,title,var,df_augmented,df_original,f_name)



ylabel = 'Validation PPL'
xlabel='Epoch'
title = 'Validation PPL  vs. Epoch'
var = 'valperp'
f_name ='valperp-epoch.pdf'

plot_linear(xlabel,ylabel,title,var,df_augmented,df_original,f_name)


ylabel = 'Generalization Gap'
xlabel='Epoch'
title = 'Validation PPL  vs. Epoch'
var = 'gen_gap'
f_name ='gengap-epoch.pdf'

plot_linear(xlabel,ylabel,title,var,df_augmented,df_original,f_name)


xlabel = 'valperp'
ylabel = 'attack1'
hue = 'hue'
legendtitle = 'Fine-tuning Method'
title= 'MIA Recall vs Validation PPL'
f_name ='mia-valppl-epoch.pdf'


plot_scatter(xlabel,ylabel,title,hue,legendtitle,df_all,f_name)