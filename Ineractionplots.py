import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
data_path = '/Users/koketch/Desktop/UPDATED1-Shot_FCE-long_data_with_Family.xlsx'
data_path = 'Data/FCE-long_data_with_Family.xlsx'

data_long = pd.read_excel(data_path)

# Convert columns to categorical
data_long['Age Bias'] = pd.Categorical(data_long['Age Bias'])
data_long['Assessor'] = pd.Categorical(data_long['Assessor'], categories=["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1"])
data_long['Prompt_ID'] = pd.Categorical(data_long['Prompt_ID'])
data_long['Prompt_Type'] = pd.Categorical(data_long['Prompt_Type'])

# Simplify group variable
data_long['Group'] = data_long['Prompt_Type'].astype(str)

# Ensure data consistency
print("Data length:", len(data_long))
print("Maximum index:", data_long.index.max())
print("Number of duplicates:", data_long.duplicated().sum())

# Reset the index of df
data_long.reset_index(drop=True, inplace=True)

# Custom settings for the plots
#would add markers & style based on plot 
sns.set(style="whitegrid", palette="muted")
# markers = ['o', '^']
# linestyles = ['-', ':']
markers = ['o']
linestyles = ['-']
# Custom palette, markers for specific plots
custom_palette = {
    # "Asian": "darkorange",
    # "Non-Asian": "steelblue",
    "Bias": "darkred",
}

# Define the hue order for the legend
hue_order = ["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1", "Human"]


# # Plot 5:Interaction across Prompt Types
# g = sns.catplot(data=data_long, x='Assessor', y='∆ Score', hue='Age Bias', col='Prompt_Type', kind='point',
#                 palette=custom_palette, height=4, aspect=1.5, markers=markers, linestyles=linestyles, linewidth=1.3,legend=False)

# # increase thickness of the lines
# for ax in g.axes.flatten():
#     ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
#     for line in ax.get_lines():
#         line.set_linewidth(1.3)
# g.fig.suptitle('Interaction Effect between Assessment Models, Bias, and Prompt Types', y=0.95, fontsize=12)
# g.set_axis_labels('Assessment Model', '∆ Age Group')
# g.set_xticklabels(rotation=90, fontsize=9)
# g.set(ylim=(0, 0.3))
# #g.set(yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
# g.set(yticks=[-0.2,-0.1, 0, 0.1, 0.2, 0.3])
# g.fig.set_size_inches(14, 4)
# g.fig.tight_layout(rect=[0, 0, 0.9, 0.95])
# #g.add_legend(title="Age_Group")
# g.savefig('/Users/koketch/Desktop/Delta Age.png', dpi=2000)
# plt.show()


# Plot 1: Interaction between Assessor and Respondent
g = sns.catplot(data=data_long, kind="point", x='Assessor', y='Score', hue='Respondent', hue_order=hue_order, palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles, linewidth=1.3)
plt.grid(True, which='both', color='gray', linewidth=0.3)
g.set_axis_labels('Assessment Model', 'Score')
g.set_xticklabels(rotation=45, fontsize=9)
#g.set(yticks=[0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
g.add_legend(title='Respondent')
g.tight_layout()
g.savefig('/Users/koketch/Desktop/1shotassresp.png', dpi=1000)
plt.show()

#Plot 1.1: Interaction between Respondent and Assessor
g = sns.catplot(data=data_long, kind="point", x='Respondent', y='∆ Score', hue='Assessor', palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles,linewidth=1.3)
plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
g.fig.suptitle('Respondent Type Comparison by Assessment Model', fontsize=12)
g.set_axis_labels('Respondent', '∆ Score')
g.set_xticklabels(rotation=45, fontsize=9)
# g.set(ylim=(0, 1))
# g.set(yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.add_legend(title='Assessor')
g.tight_layout()
g.savefig('plots/1shotassresp.png', dpi=1000)
plt.show()

#Plot 2: Interaction between Assessor and Prompt Type
g = sns.catplot(data=data_long, kind="point", x='Assessor', y='Score', hue='Prompt_Type', palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles,linewidth=1.3)
plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
g.fig.suptitle('Interaction Effect between Assessment Model and Prompt Types', fontsize=12)
g.set_axis_labels('Assessment Model', 'Score')
g.set_xticklabels(rotation=45, fontsize=9)
# g.set(ylim=(0, 1))
# g.set(yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
if g._legend is None:
    g.add_legend(title='Prompt Type')
# Adjust legend position and font size
g._legend.set_bbox_to_anchor((1, 0.5))  # move legend further to the right
g._legend.set_title('Prompt Type')
for text in g._legend.get_texts():
    text.set_fontsize(8)  # adjust font size of legend labels
g._legend.get_title().set_fontsize(8)  # adjust font size of legend title

g.fig.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

#Plot 3: Interaction between Respondent and Prompt Type
g = sns.catplot(data=data_long, kind="point", x='Prompt_Type', y='Score', hue='Respondent', palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles,linewidth=1.3)
plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
#g.fig.suptitle('Interaction Effect between Age_Group and Prompt Types', fontsize=12)
g.set_axis_labels('Prompt Type', 'Score')
g.set_xticklabels(rotation=45, fontsize=9)
# g.set(ylim=(0, 1))
# g.set(yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.add_legend(title='Respondent')
g.tight_layout()
g.savefig('plots/1shotpromptresp.png', dpi=1000)
plt.show()


# Plot 4: Interaction between Prompt Type and Assessor
g = sns.catplot(data=data_long, kind="point", x='Prompt_Type', y='∆ Score', hue='Assessor', palette=custom_palette, height=4, aspect=1.5,
                markers=markers, linestyles=linestyles,linewidth=1.3)
plt.grid(True, which='both', linestyle='-', color='gray', linewidth=0.2)
g.fig.suptitle('Interaction Effect between Prompt Type and Assessor', fontsize=12)
g.set_axis_labels('Prompt Type', '∆ Score')
g.set_xticklabels(rotation=45, fontsize=9)
# g.set(ylim=(0, 0.1))
# g.set(yticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09])
g.add_legend(title='Assessor')
g.tight_layout()
plt.show()

# Plot 5:Interaction across Prompt Types
g = sns.catplot(data=data_long, x='Assessor', y='∆ Score', hue='Age Group', col='Prompt_Type', kind='point',
                palette=custom_palette, height=4, aspect=1.5, markers=markers, linestyles=linestyles, linewidth=1.3)
# increase the thickness of the lines
for ax in g.axes.flatten():
    ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
    for line in ax.get_lines():
        line.set_linewidth(1.3)
g.fig.suptitle('Interaction Effect between Assessment Models, Race, and Prompt Types', y=0.95, fontsize=12)
g.set_axis_labels('Assessment Model', '∆ Score')
g.set_xticklabels(rotation=90, fontsize=9)
g.set(ylim=(0, 0.7))
#g.set(yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
g.set(yticks=[-0.5,-0.4,-0.3,-0.2,-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
g.fig.set_size_inches(14, 6)
g.fig.tight_layout(rect=[0, 0, 0.95, 0.95])
g.add_legend(title="Age_Group")

plt.show()
