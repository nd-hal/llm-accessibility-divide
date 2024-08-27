import matplotlib.pyplot as plt

# Data
models = ['GPT-3.5', 'GPT-4', 'GPT-4 Omni', 'Llama-2', 'Llama-3', 'Llama-3.1']
input_costs = [116.7109, 2337.9695, 390.5797, 50.5953, 50.5323, 250.2109]
output_costs = [21.4, 1012.11, 296.63, 41.0691, 38.9646, 20.7449]

# Converting costs to smaller units for clarity
input_costs_usd = [x / 1 for x in input_costs]
output_costs_usd = [x / 1 for x in output_costs]

# Plot
fig, ax = plt.subplots()
ax.plot(models, input_costs_usd, marker='o', linestyle='-', color='steelblue', label='Input')
ax.plot(models, output_costs_usd, marker='o', linestyle='--', color='darkorange', label='Output')

# Title and labels
#ax.set_title('Comparison of Input and Output Costs for Various Language Models', fontsize=10)
ax.set_ylabel('Cost in Thousands (USD)', fontsize=10, fontweight='bold')
ax.set_xlabel('Model', fontsize=10, fontweight='bold')

# Set y-axis limit
ax.set_ylim(0, 2500)

# Legend
ax.legend()

# Show plot
plt.xticks(rotation=0, fontsize=8)  # Rotate model names for better visibility
plt.yticks(fontsize=8)
plt.grid(True)
plt.show()
