import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


## STEP 1: generate what is currently figure 4 in TACL 
# two-panel plot with one legend at the bottom

def generate_fig4():

    # Load data
    data_path = './Data/1-ShotHuman-LLMslong_data.xlsx'  
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
        dodge=0.4,
        markersize=5,
        alpha=0.5
    )

    #plt.grid(True, which='both', color='gray', linewidth=0.3)
    g.add_legend(title='Respondent')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=7, title='Respondent', frameon=False,
    )

    g.savefig('plots/1shotplot.png', dpi=1000)
    #plt.show()

def generate_fig5():
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
    
    #g.add_legend(title='Age Group')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=7, frameon=False,
    )

    # Customize the plot further 
    g.set_axis_labels('Assessment Model', '∆ Score')
    #g.set_xticklabels(fontsize=9)
    g.set(ylim=(-0.1, 0.15)) 
    g.set(yticks=[-0.1,-0.05, 0, 0.05, 0.10, 0.15])
    g.savefig('plots/interactionsAge.png', dpi=1000)
    #plt.show()

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
    
    #g.add_legend(title='Race')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, 1), ncol=7, frameon=False,
    )

    # Customize the plot further 
    g.set_axis_labels('Assessment Model', '∆ Score')
    #g.set_xticklabels(fontsize=9)
    g.set(ylim=(-0.1, 0.15)) 
    g.set(yticks=[-0.1,-0.05, 0, 0.05, 0.10, 0.15])
    g.savefig('plots/interactionsRace.png', dpi=1000)
    #plt.show()

    # Figure 5:Interaction across Prompt Types
    # JL: I've commented out the below as we aren't using it at the moment...
    markers = ["o", "^", "s", "p", "*", "h", "D"]
    linestyles = ["-", "--", "-.", ":", "-", "-.", ":"]

    # Hue order for the legend
    hue_order = ["GPT3.5", "GPT4", "GPT4o", "Llama2", "Llama3", "Llama3.1", "Human"]
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
    g.fig.suptitle('Interaction Effect between Assessment Models, Age, and Prompt Types', y=0.95, fontsize=12)
    g.set_axis_labels('Assessment Model', '∆ Score')
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=90, fontsize=9)
    g.set(ylim=(-0.2, 0.3))
    g.set(yticks=[-0.2,-0.1, 0, 0.1, 0.2, 0.3])
    g.fig.set_size_inches(14, 4)
    g.fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    # g.add_legend(title="Age_Group")

    #g.add_legend(title='Race')
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.1), ncol=7, frameon=False,
    )
    g.savefig('plots/interactionsPromptAge.png', dpi=1000)
    #plt.show()

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
    g.fig.suptitle('Interaction Effect between Assessment Models, Race, and Prompt Types', y=0.95, fontsize=12)
    g.set_axis_labels('Assessment Model', '∆ Score')
    g.set_xticklabels(rotation=90, fontsize=9)
    g.set(ylim=(-0.2, 0.3))
    g.set_titles("{col_name}")
    g.set(yticks=[-0.2,-0.1, 0, 0.1, 0.2, 0.3])
    g.fig.set_size_inches(14, 4)
    g.fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    # g.add_legend(title="Age_Group")
    sns.move_legend(
        g, "lower center",
        bbox_to_anchor=(.5, -0.1), ncol=7, frameon=False,
    )
    g.savefig('plots/interactionsPromptRace.png', dpi=1000)
    #plt.show()


generate_fig4()
#generate_fig5()
# Update # Add a small change
