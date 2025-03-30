import matplotlib.pyplot as plt
import pandas as pd
import glob

# Get all CSV files from reports directory
csv_files = glob.glob('reports/*.csv')

# Define colors for the plots
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# Store means for all columns for each file
all_means = {}
labels = []

for file in csv_files:
    # Read the CSV file
    data = pd.read_csv(file)
    
    # Only process files with 10 rows
    if len(data) == 10:
        # Calculate mean for all numeric columns, excluding the first column
        means = data.iloc[:, 1:].mean(numeric_only=True)  # Skip the first column
        all_means[file] = means
        labels.append(file)

# Get metrics from the columns (excluding the first column which is Iteration)
metrics = list(all_means[labels[0]].index) if labels else []

# Calculate number of rows and columns for subplots
n_metrics = len(metrics)
n_cols = 2
n_rows = (n_metrics + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
axes = axes.flatten()

# Create a bar plot for each metric
for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Get values for this metric from all files
    values = [all_means[file][metric] for file in labels]
    
    # Create bar plot
    ax.bar(range(len(values)), values, color=colors[:len(values)])
    
    # Create shorter labels
    shortened_labels = []
    for label in labels:
        parts = label.split('/')[-1].split('_')
        compiler = parts[1]
        opt_status = "Opt" if "optimizado" in label else "No-opt"
        shortened_labels.append(f"{compiler}\n{opt_status}")
    
    # Set labels and title
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(shortened_labels, rotation=45)
    ax.set_title(f'Average {metric}')
    ax.set_ylabel(metric)

# Remove any unused subplots
for idx in range(len(metrics), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('chart.jpg', bbox_inches='tight', dpi=300)

