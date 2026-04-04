import pandas as pd
import matplotlib.pyplot as plt

# Read the data into a Pandas DataFrame
df = pd.read_csv("/Users/itak/Documents/CMU/24775_Bioinspired_Robot/MeowMentum/telemetry/telemetry_1775333675.csv")

# Create the plot
plt.figure(figsize=(10, 6))

# Loop through all columns except 'Time' to plot them
for col in df.columns:
    if col != 'Time':
        plt.plot(df['Time'], df[col], label=col)

# Add labels and styling
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.title('Cat Robot Kinematics/Commands over Time')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()