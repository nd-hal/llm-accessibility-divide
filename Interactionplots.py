import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_path = 'Data/1-ShotHuman-LLMslong_data.xlsx'  

data_long = pd.read_excel(data_path)

# Convert columns to categorical
data_long['Assessor'] = pd.Categorical(data_long['Assessor'], categories=["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1"])
data_long['Prompt_ID'] = pd.Categorical(data_long['Prompt_ID'])
data_long['Prompt_Type'] = pd.Categorical(data_long['Prompt_Type'])

# Simplify group variable
data_long['Group'] = data_long['Prompt_Type'].astype(str)

# Remove duplicates if any
data_long.drop_duplicates(inplace=True)

# Reset the index of df
data_long.reset_index(drop=True, inplace=True)

# Define a custom color palette using color names
custom_palette = {
    "GPT3.5": "steelblue",  
    "GPT4": "steelblue",   
    "GPT4o": "steelblue",   
    "Llama2": "darkorange",   
    "Llama3": "darkorange",   
    "Llama3.1": "darkorange", 
    "Human": "olive"        
}

# Custom settings for the plots
sns.set(style="whitegrid", palette="muted")
markers = ["o", "^", "s", "p", "*", "h", "D"]
linestyles = ["-", "--", "-.", ":", "-", "-.", ":"]

# Hue order for the legend
hue_order = ["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1", "Human"]

# Figure 3a: Interaction between Assessor and Respondent
g = sns.catplot(data=data_long, kind="point", x='Assessor', y='Score', hue='Respondent', hue_order=hue_order, palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles, linewidth=1.3)
plt.grid(True, which='both', color='gray', linewidth=0.3)
g.set_axis_labels('Assessment Model', 'Score')
g.set_xticklabels(rotation=45, fontsize=9)
g.add_legend(title='Respondent')
g.tight_layout()
g.savefig('/Users/koketch/Desktop/1shotassresp.png', dpi=1000)
plt.show()

# Figure 3b: Interaction between Respondent and Prompt Type
g = sns.catplot(data=data_long, kind="point", x='Prompt_Type', y='Score', hue='Respondent', hue_order=hue_order, palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles, linewidth=1.3)
plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
g.set_axis_labels('Prompt Type', 'Score')
g.set_xticklabels(rotation=45, fontsize=9)
g.add_legend(title='Respondent')
g.tight_layout()
g.savefig('/Users/koketch/Desktop/1shotpromptresp.png', dpi=1000)
plt.show()


###FAIRNESS/BIAS PLOTS###
# Load data
data_path = 'Data/FCE-long_data_with_Family.xlsx'
data_long = pd.read_excel(data_path)

# Convert columns to categorical
data_long['Age Group'] = pd.Categorical(data_long['Age Group'])
data_long['Assessor'] = pd.Categorical(data_long['Assessor'], categories=["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1"])
data_long['Prompt_ID'] = pd.Categorical(data_long['Prompt_ID'])
data_long['Prompt_Type'] = pd.Categorical(data_long['Prompt_Type'])
data_long['Group'] = data_long['Prompt_Type'].astype(str)

# Reset the index of df
data_long.reset_index(drop=True, inplace=True)

# Custom palette, markers
custom_palette = {
    "Old": "darkorange",
    "Young": "steelblue",
}

# Figure 4a: Interaction across Assessment Models and Age Groups
g = sns.catplot(data=data_long, x='Assessor', y='∆ Score', hue='Age Group', kind='point',
                palette=custom_palette, height=4, aspect=1.5, linestyles=["-", "--"], markers=["o", "^"])

# Customize the plot further 
g.set_axis_labels('Assessment Model', '∆ Score')
g.set_xticklabels(rotation=45, fontsize=9)
g.set(ylim=(-0.1, 0.15)) 
g.set(yticks=[-0.1,-0.05, 0, 0.05, 0.10, 0.15])
g.add_legend(title="Age Group")
g.despine(left=True)
plt.grid(True, which='both', linestyle='-', color='gray', linewidth=1.3)
plt.show()

# Figure 5:Interaction across Prompt Types
g = sns.catplot(data=data_long, x='Assessor', y='∆ Score', hue='Age Group', col='Prompt_Type', kind='point',
                palette=custom_palette, height=4, aspect=1.5, markers=markers, linestyles=linestyles, linewidth=1.3)
# increase the thickness of the lines
for ax in g.axes.flatten():
    ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
    for line in ax.get_lines():
        line.set_linewidth(1.3)
g.fig.suptitle('Interaction Effect between Assessment Models, Age, and Prompt Types', y=0.95, fontsize=12)
g.set_axis_labels('Assessment Model', '∆ Score')
g.set_xticklabels(rotation=90, fontsize=9)
g.set(ylim=(-0.2, 0.3))
g.set(yticks=[-0.2,-0.1, 0, 0.1, 0.2, 0.3])
g.fig.set_size_inches(14, 4)
g.fig.tight_layout(rect=[0, 0, 0.95, 0.95])
# g.add_legend(title="Age_Group")
plt.show()
