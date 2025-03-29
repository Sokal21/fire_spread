import matplotlib.pyplot as plt
import pandas as pd
import glob

# Assuming the CSV files are in the 'reports' directory
csv_files = glob.glob('reports/*.csv')

# Define a color map for the files
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i, file in enumerate(csv_files):
    # Read the CSV file
    data = pd.read_csv(file)
    
    # Assuming the CSV has a column named 'IPS' for performance
    ips = data['IPS'].to_numpy()  # Convert to numpy array
    
    # Generate bar plot
    x_positions = range(len(ips))
    width = 0.35  # Width of the bars
    plt.bar([x + i*width for x in x_positions], ips, width, 
            color=colors[i % len(colors)], label=file)
    
# Add labels and title
plt.xlabel('Index')
plt.ylabel('IPS')
plt.title('Performance based on IPS')
plt.legend()

# Set x-axis ticks to show integer indices
plt.xticks([x + width/2 for x in x_positions], [str(x) for x in x_positions])

# Save the plot to a file
plt.savefig('chart.jpg')

