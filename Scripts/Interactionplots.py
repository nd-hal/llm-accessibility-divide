import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


## STEP 1: generate what is currently figure 3 
# two-panel plot with one legend at the bottom

def generate_fig5():

    # Load data
    data_path = '.Data/New 1-ShotHuman-LLMsLongData.xlsx' 
    data_long = pd.read_excel(data_path)

    # Convert columns to categorical
    data_long['Assessor'] = pd.Categorical(data_long['Assessor'], categories=["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1",'Deepseek-R1', 'Qwen2.5'])
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
        "Deepseek-R1": "darkorange",
        "Qwen2.5": "darkorange",
        #"Olmo2": "green",
        "Human": "olive"        
    }

    # Custom settings for the plots
    sns.set(style="whitegrid", palette="muted")
    markers = ["o", "^", "s", "p", "*", "h","X","H","d","D"]
    linestyles = ["-", "--", "-.", ":", "-", "-.","-","--", "-.",":"]

    # Hue order for the legend
    hue_order = ["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1",'Deepseek-R1','Qwen2.5', "Human"]

    
    # Figure 5: Interaction between Assessor and Respondent
    sns.set_style("ticks")

    
    g = sns.PairGrid(
        data_long, 
        y_vars=["Score"],
        x_vars=["Assessor", "Prompt_Type"],
        hue='Respondent', 
        hue_order=hue_order, 
        palette=custom_palette, 
        height=4, 
        aspect=1.5
    )
    g.map(
        sns.pointplot,
        markers=markers, 
        linestyles=linestyles, 
        linewidth=1.3,
        dodge=0.5,
        markersize=5,
        alpha=1
    )
   
    handles, labels = g.axes[0][0].get_legend_handles_labels()

    plt.legend(
        handles, labels, title="Respondent",
        loc='upper center', bbox_to_anchor=(0, 1.18), 
        ncol=len(hue_order), frameon=False,fontsize=12
    )
    plt.subplots_adjust(top=0.9)  # Moves figure down to make space

    # Reduce x-axis font size
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=10, fontsize=10)
        ax.set_yticks(np.arange(0.5, 0.9, 0.1))  

    plt.show()

    #plt.grid(True, which='both', color='gray', linewidth=0.3)
    g.add_legend(title='Respondent')
    sns.move_legend(
        g, "upper center",
        bbox_to_anchor=(.5, 2), ncol=len(hue_order), title='Respondent', frameon=False,
    )
     #Reduce x-axis font size
    for ax in g.axes.flatten():
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)

    #g.savefig('/Users/koketch/Desktop/LLMs_Accessibility_Divide/Results/1shotplot.png', dpi=2000)
    plt.show()


def generate_fig3():
    ###FAIRNESS/BIAS PLOTS###
    # Load data
    data_path = '.Data/NEW-1-Shot_FCE-long_data_with_Age-Race.xlsx'
    data_long = pd.read_excel(data_path)

    # Convert columns to categorical
    data_long['Age Group'] = pd.Categorical(data_long['Age Group'])
    data_long['Assessor'] = pd.Categorical(data_long['Assessor'], categories=["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1",'Deepseek-R1', 'Qwen2.5'])
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

    # Figure 3b: Interaction across Assessment Models and Age Groups
    sns.set_style("ticks")

    g = sns.catplot(
        data=data_long, 
        x='Assessor', 
        y='∆ Score', 
        hue='Age Group', 
        kind='point',
        palette=custom_palette, 
        height=4, 
        aspect=1.25,
        linestyles=["-", "--"], 
        markers=["o", "^"],
        linewidth=1.3
    )
    g.fig.subplots_adjust(bottom=0.3)  # Avoid excessive space
    #g.add_legend(title='Age Group')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.05), ncol=9, frameon=False,
    )

    # Customize the plot further 
    g.set_axis_labels('Assessment Model', '∆ Score')
    #g.set_xticklabels(fontsize=9)
    g.set(ylim=(-0.1, 0.15)) 
    g.set(yticks=[-0.1,-0.05, 0, 0.05, 0.10, 0.15])
    #g.savefig('plots/interactionsAge.png', dpi=1000)
    plt.show()

    # Custom palette, markers
    custom_palette = {
        "Non-Asian": "darkorange",
        "Asian": "steelblue",
    }
    
    g = sns.catplot(
        data=data_long, 
        x='Assessor', 
        y='∆ Score', 
        hue='Race', 
        kind='point',
        palette=custom_palette, 
        height=4, 
        aspect=1.25,
        linestyles=["-", "--"], 
        markers=["o", "^"],
        linewidth=1.3
    )
    g.fig.subplots_adjust(bottom=0.25)

    #g.add_legend(title='Race')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.05), ncol=9, frameon=False,
    )


    # Customize the plot further 
    g.set_axis_labels('Assessment Model', '∆ Score')
    g.set_xticklabels(fontsize=12)
    g.set(ylim=(-0.1, 0.15)) 
    g.set(yticks=[-0.1,-0.05, 0, 0.05, 0.10, 0.15])
    #g.savefig('plots/interactionsRace.png', dpi=1000)
    plt.show()

    # Figure 3:Interaction across Prompt Types
    # JL: I've commented out the below as we aren't using it at the moment...
    
    markers = ["o", "^", "s", "p", "*", "h","X","H","D"]
    linestyles = ["-", "--", "-.", ":", "-", "-.","-","--",":"]

    # Hue order for the legend
    hue_order = ["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1",'Deepseek-R1', 'Qwen2.5', "Human"]
    # Custom palette, markers
    custom_palette = {
        "Old": "darkorange",
        "Young": "steelblue",
    }

    g = sns.catplot(data=data_long, x='Assessor', y='∆ Score', hue='Age Group', col='Prompt_Type', kind='point',
                    palette=custom_palette, height=4, aspect=1.5, markers=markers, linestyles=linestyles, linewidth=1.3)
    # increase the thickness of the lines
    for ax in g.axes.flatten():
        ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
        for line in ax.get_lines():
            line.set_linewidth(1.3)
    g.fig.suptitle('Interaction Effect between Assessment Models, Age, and Prompt Types', y=0.95, fontsize=11)
    g.set_axis_labels('Assessment Model', '∆ Score')
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=90, fontsize=10)
    g.set(ylim=(-0.2, 0.3))
    g.set(yticks=[-0.2,-0.1, 0, 0.1, 0.2, 0.3])
    g.fig.set_size_inches(14, 4)
    g.fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    # g.add_legend(title="Age_Group")
    g.fig.subplots_adjust(bottom=0.4)  # Avoid excessive space
    #g.add_legend(title='Race')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.02), ncol=9, frameon=False,
    )
    
    #g.savefig('/Users/koketch/Desktop/LLMs_Accessibility_Divide/Results/interactionsPromptAge.png', dpi=2000)
    plt.show()

    # Custom palette, markers
    custom_palette = {
        "Non-Asian": "darkorange",
        "Asian": "steelblue",
    }

    g = sns.catplot(data=data_long, x='Assessor', y='∆ Score', hue='Race', col='Prompt_Type', kind='point',
                    palette=custom_palette, height=4, aspect=1.5, markers=markers, linestyles=linestyles, linewidth=1.3)
    # increase the thickness of the lines
    for ax in g.axes.flatten():
        ax.grid(True, which='both', linestyle='-', color='gray', linewidth=0.3)
        for line in ax.get_lines():
            line.set_linewidth(1.3)
    g.fig.suptitle('Interaction Effect between Assessment Models, Race, and Prompt Types', y=0.95, fontsize=11)
    g.set_axis_labels('Assessment Model', '∆ Score')
    g.set_xticklabels(rotation=90, fontsize=10)
    g.set(ylim=(-0.2, 0.3))
    g.set_titles("{col_name}")
    g.set(yticks=[-0.2,-0.1, 0, 0.1, 0.2, 0.3])
    g.fig.set_size_inches(14, 4)
    g.fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    # g.add_legend(title="Age_Group")
    g.fig.subplots_adjust(bottom=0.4)
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.02), ncol=9, frameon=False,
    )
    #g.fig.subplots_adjust(top=0.5)  # Increase space below the figure

    g.savefig('/Users/koketch/Desktop/LLMs_Accessibility_Divide/Results/interactionsPromptRace.png', dpi=2000)
    plt.show()


#generate_fig5()
generate_fig3()