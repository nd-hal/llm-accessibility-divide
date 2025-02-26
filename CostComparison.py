import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext

# Data
models = ['GPT-3.5', 'GPT-4', 'GPT-4 Omni', 'Llama 2', 'Llama 3', 'Llama 3.1', 'DeepSeek-R1', 'Qwen2.5']
input_costs = [116.7109, 2337.9695, 390.5797, 50.5953, 50.5323, 250.2109, 36.2997, 6.2919]
output_costs = [21.4, 1012.11, 296.63, 41.0691, 38.9646, 20.7449, 51.1891, 8.5315]

fig, ax = plt.subplots(figsize=(6, 4))

# Plot
ax.plot(models, input_costs, marker='o', linestyle='-', color='steelblue', label='Input Cost')
ax.plot(models, output_costs, marker='o', linestyle='--', color='darkorange', label='Output Cost')

# Hide spines
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_bounds(0, len(models))

# Labels
ax.set_ylabel('Cost (USD)')
ax.set_xlabel('Model')
ax.set_xticks(range(len(models)))  # Set tick positions
ax.set_xticklabels(models, rotation=45, ha='right')  # Rotate labels

# Use log scale and properly set ticks
ax.set_yscale('log')
ax.yaxis.set_major_formatter(LogFormatterMathtext())  # Automatically format labels

# Add legend in the upper right
ax.legend(frameon=False, fontsize=10, loc='upper right', bbox_to_anchor=(1, 1))

plt.tight_layout() 

# Save plot
#plt.savefig('.Results/costv2.png', dpi=1000)
plt.show()
