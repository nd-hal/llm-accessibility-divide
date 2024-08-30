import matplotlib.pyplot as plt


# This file generates what is currently figure 2 in the tacl paper.
# specifically, input and output token cost across LLMs

# Data
models = ['GPT-3.5', 'GPT-4', 'GPT-4 Omni', 'Llama-2', 'Llama-3', 'Llama-3.1']
input_costs = [116.7109, 2337.9695, 390.5797, 50.5953, 50.5323, 250.2109]
output_costs = [21.4, 1012.11, 296.63, 41.0691, 38.9646, 20.7449]

# Converting costs to smaller units for clarity
# JL: set the values as their true values, we deal with scaling later
input_costs_usd = [x * 1000 for x in input_costs]
output_costs_usd = [x * 1000 for x in output_costs]

# Plot
fig, ax = plt.subplots(figsize=(6, 4))


# Formatting
# source: https://www.practicaldatascience.org/notebooks/class_5/week_5/46_making_plots_pretty.html

# Define font sizes
SIZE_DEFAULT = 12
SIZE_LARGE = 20
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_LARGE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_LARGE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_LARGE)  # fontsize of the tick labels


ax.plot(models, input_costs_usd, marker='o', linestyle='-', color='steelblue', label='Input')
ax.plot(models, output_costs_usd, marker='o', linestyle='--', color='darkorange', label='Output')

# Redo legend

# Plot the cost text
ax.text(
    5 * 1.02,
    input_costs_usd[-1],
    "Inputs",
    color="steelblue",
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)

ax.text(
    5 * 1.02,
    output_costs_usd[-1],
    "Outputs",
    color="darkorange",
    fontweight="bold",
    horizontalalignment="left",
    verticalalignment="center",
)


# Hide the all but the bottom spines (axis lines)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(0, len(models))


# Title and labels
#ax.set_title('Comparison of Input and Output Costs for Various Language Models', fontsize=10)
ax.set_ylabel('Cost (log USD)', fontweight='bold')
ax.set_xlabel('Model', fontweight='bold')

# Set y-axis limit
#ax.set_ylim(0, 2500)
ax.set_yscale('log')

# Remove Legend
#ax.legend()

# Show plot
plt.xticks()  # Rotate model names for better visibility
plt.yticks()
# remove grid
#plt.grid(True)
plt.savefig('plots/costv2.png', dpi=1000)
#plt.show()
