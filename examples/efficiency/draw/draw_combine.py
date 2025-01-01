import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# 使用 seaborn 的美化主题
sns.set_theme(style="whitegrid")

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def process_and_plot(linear_file_paths, quadratic_file_paths, names, ranges):
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']  # Predefined color list
    fig, axes = plt.subplots(1, 2, figsize=(4.7142857, 3), sharex='col')  # Add sharey=True to share y-axis

    # Linear regression plotting
    for idx, file_path in enumerate(linear_file_paths):
        # Read the txt file
        with open(file_path, 'r') as file:
            data = file.read()

        # Extract sequence lengths and times using regex
        seq_lens = [1024 * int(x) for x in re.findall(r'packing num = (\d+):', data)]
        times = [float(x) for x in re.findall(r'(\d+\.\d+)s', data)]

        seq_lens = seq_lens[:int(ranges[idx] * len(seq_lens))]
        times = times[:int(ranges[idx] * len(times))]

        # Convert to numpy arrays
        X = np.array(seq_lens).reshape(-1, 1)
        y = np.array(times)

        # Perform Linear Regression (first subplot)
        linear_model = LinearRegression(fit_intercept=False)  # No intercept
        linear_model.fit(X, y)
        linear_coefficient = linear_model.coef_[0]
        print(f'Linear coefficient for file {file_path}: {linear_coefficient}')

        # Generate linear regression data
        X_fit = np.linspace(min(seq_lens), max(seq_lens), 100).reshape(-1, 1)
        y_fit_linear = linear_model.predict(X_fit)

        # Plot linear regression in the first subplot
        axes[1].scatter(seq_lens, times, color=colors[idx % len(colors)], alpha=0.6)
        axes[1].plot(X_fit, y_fit_linear, color=colors[idx % len(colors)], label=names[idx])

    # Quadratic regression plotting
    for idx, file_path in enumerate(quadratic_file_paths):
        # Read the txt file
        with open(file_path, 'r') as file:
            data = file.read()

        # Extract sequence lengths and times using regex
        seq_lens = [int(x) for x in re.findall(r'seq len = (\d+):', data)]
        times = [float(x) for x in re.findall(r'(\d+\.\d+)s', data)]

        seq_lens = seq_lens[:int(ranges[idx] * len(seq_lens))]
        times = times[:int(ranges[idx] * len(times))]

        # Convert to numpy arrays
        X = np.array(seq_lens).reshape(-1, 1)
        y = np.array(times)

        # Perform Quadratic Regression (second subplot)
        X_quad = X**2
        quad_model = LinearRegression(fit_intercept=False)
        quad_model.fit(X_quad, y)
        quad_coefficient = quad_model.coef_[0]
        print(f'Quadratic coefficient for file {file_path}: {quad_coefficient}')

        # Generate quadratic regression data
        X_fit = np.linspace(min(seq_lens), max(seq_lens), 100).reshape(-1, 1)
        X_fit_quad = X_fit**2
        y_fit_quad = quad_model.predict(X_fit_quad)

        # Plot quadratic regression in the second subplot
        axes[0].scatter(seq_lens, times, color=colors[idx % len(colors)], alpha=0.6)
        axes[0].plot(X_fit, y_fit_quad, color=colors[idx % len(colors)], label=names[idx])

    # Customize the first subplot (Linear Regression)
    # fig.supxlabel('Sequence Length (token)') 
    # fig.supylabel('Time (s)') 
    axes[0].set_ylabel('Time (s)')
    fig.text(0.5, 0.09, 'Sequence Length', ha='center', va='center')
    axes[1].set_title('Varlen Attn', fontweight='bold')

    # Customize the second subplot (Quadratic Regression)
    axes[0].set_title('Standard Attn', fontweight='bold')
    
    axes[0].grid(visible=True, linestyle='--')
    axes[1].grid(visible=True, linestyle='--')
    
    # 获取图例句柄和标签（从第一个子图中获取）
    handles, labels = axes[0].get_legend_handles_labels()
    # 添加全局图例，放在图的正下方
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, fancybox=True)

    # Shared title and layout adjustments
    plt.tight_layout(rect=[0, 0.08, 1, 0.85])
    plt.savefig("./fig_combined.png")
    plt.savefig("./varlen_attn.svg", pad_inches=0.01, bbox_inches="tight")
    plt.show()

# Example usage with two different sets of file paths
linear_file_paths = ['./tencent_packing/num_heads_4.txt', './tencent_packing/num_heads_8.txt', './tencent_packing/num_heads_16.txt', './tencent_packing/num_heads_32.txt']
quadratic_file_paths = ['./tencent_padding/num_heads_4.txt', './tencent_padding/num_heads_8.txt', './tencent_padding/num_heads_16.txt', './tencent_padding/num_heads_32.txt']

ranges = [1, 1, 1, 1]  # Same for both sets
names = ["TP=8", "TP=4", "TP=2", "TP=1"]  # Shared labels
process_and_plot(linear_file_paths, quadratic_file_paths, names, ranges)
