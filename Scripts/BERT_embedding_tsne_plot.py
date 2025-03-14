
# !pip install transformers

##Human & LLMs###

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load the embeddings
path_to_embeddings = './Data/Essay_embeddings2.pt'
essay_embeddings = torch.load(path_to_embeddings)

# Define the full list of sources and their counts
source_labels = ['Human','GPT-3.5', 'GPT-4', 'GPT-4o', 'Llama-2', 'Llama-3', 'Llama-3.1', 'DeepSeek-R1','Qwen2.5','Olmo2']
source_counts = [15445, 1537, 1487, 1527, 1537, 1537, 1537, 1537, 1537, 1537]

# Generate sources list based on actual counts
sources = []
for label, count in zip(source_labels, source_counts):
    sources.extend([label] * count)

# Define custom colors and markers for each source
custom_styles = {

    'Human': {'color': 'orange', 'marker': '^'},
    'GPT-3.5': {'color': 'blue', 'marker': '^'},
    'GPT-4': {'color': 'red', 'marker': '^'},
    'GPT-4o': {'color': 'aqua', 'marker': '^'},
    'Llama-2': {'color': 'lime', 'marker': '^'},
    'Llama-3': {'color': 'steelblue', 'marker': '^'},
    'Llama-3.1': {'color': 'green', 'marker': '^'},
    'DeepSeek-R1': {'color': 'black', 'marker': '^'},
    'Qwen2.5': {'color': 'deeppink', 'marker': '^'},
    'Olmo2': {'color': 'indigo', 'marker': '^'}
}

# Apply t-SNE
def apply_tsne(embeddings, initial_perplexity=40):
    if embeddings is not None and embeddings.size(0) > 1:  # Ensure more than one sample
        perplexity = min(embeddings.size(0) - 1, initial_perplexity)
        tsne_model = TSNE(n_components=2, perplexity=perplexity, random_state=123)
        return tsne_model.fit_transform(embeddings.detach().cpu().numpy())
    else:
        return None

# Apply t-SNE to all embeddings
reduced_embeddings = apply_tsne(essay_embeddings)

# Mapping to rename labels for display
label_mapping = {'Olmo2': 'OLMo 2'}



# Visualization function
def plot_tsne(reduced_embeddings, sources, title, fontsize=20):
    if reduced_embeddings is not None:
        plt.figure(figsize=(14, 10))

        # Explicitly set title font size
        plt.title(title, fontsize=fontsize)

        # Explicitly set axis label font sizes
        plt.xlabel('t-SNE Component 1', fontsize=fontsize)
        plt.ylabel('t-SNE Component 2', fontsize=fontsize)

        # Plot each source with its custom style
        for source, style in custom_styles.items():
            plot_indices = [i for i, src in enumerate(sources[:len(reduced_embeddings)]) if src == source]
            display_label = label_mapping.get(source, source)
            plt.scatter(reduced_embeddings[plot_indices, 0], reduced_embeddings[plot_indices, 1],
                        c=style['color'], marker=style['marker'], label=display_label, alpha=0.8, s=8)

        # Set legend font size
        plt.legend(fontsize=fontsize - 2, loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)

        # Adjust layout to accommodate legend
        plt.tight_layout(rect=[0, 0.1, 1, 1])

    else:
        plt.text(0.5, 0.5, 'Not enough data for visualization',
                 horizontalalignment='center', fontsize=fontsize)

    # Save the plot with high resolution
    plt.savefig('./tsne.png', dpi=1000, bbox_inches='tight')

    # Show the plot
    plt.show()

# Plot all embeddings with t-SNE
plot_tsne(reduced_embeddings, sources, 't-SNE Visualization of All Essays', fontsize=20)

