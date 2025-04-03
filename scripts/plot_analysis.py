import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Load results from JSON file
with open(os.path.join('optuna_results/json/final_results.json'), 'r') as f:
    results = json.load(f)

# Define metrics and models
metrics = ["mse", "rmse", "mae", "r2"]
models = ["FullyConnectedNN", "LSTMModel", "GRUModel"]
loss_functions = ["pi", "huber", "mse", "log_cosh"]

os.makedirs("./optuna_results/plots", exist_ok=True)

# Function to plot bar charts for each metric
def plot_bar_charts(results, metric, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract values for plotting
    x_labels = []
    y_values = []
    for loss in loss_functions:
        for model in models:
            x_labels.append(f"{model} ({loss})")
            y_values.append(results[loss][model][metric])
    
    ax.barh(x_labels, y_values, color="skyblue")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Comparison of {metric.upper()} across Models and Loss Functions")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig(f"./optuna_results/plots//{filename}.png", bbox_inches="tight")
    plt.close()

# Generate bar plots for each metric
for metric in metrics:
    plot_bar_charts(results, metric, f"bar_chart_{metric}")

# Function to generate grouped bar charts
def plot_grouped_bar_charts(results, metric, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = np.arange(len(loss_functions))
    
    for i, model in enumerate(models):
        values = [results[loss][model][metric] for loss in loss_functions]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(loss_functions)
    ax.set_xlabel("Loss Function")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} Across Different Models and Loss Functions")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.savefig(f"./optuna_results/plots/grouped_bar_{metric}.png", bbox_inches="tight")
    plt.close()

# Generate grouped bar charts for each metric
for metric in metrics:
    plot_grouped_bar_charts(results, metric, f"grouped_bar_{metric}")

# Generate scatter plots with trend lines
def plot_scatter_trend(results, metric1, metric2, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_values = []
    y_values = []
    labels = []
    
    for loss in loss_functions:
        for model in models:
            x_values.append(results[loss][model][metric1])
            y_values.append(results[loss][model][metric2])
            labels.append(f"{model} ({loss})")
    
    sns.regplot(x=x_values, y=y_values, ax=ax, scatter_kws={"s": 100, "alpha": 0.7}, line_kws={"color": "red"})
    ax.set_xlabel(metric1.upper())
    ax.set_ylabel(metric2.upper())
    ax.set_title(f"{metric1.upper()} vs {metric2.upper()}")

    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))    
    plt.savefig(f"./optuna_results/plots//scatter_{metric1}_{metric2}.png", bbox_inches="tight")
    plt.close()

# Create scatter plots with trend lines
for i in range(len(metrics)):
    for j in range(i+1, len(metrics)):
        plot_scatter_trend(results, metrics[i], metrics[j], f"scatter_{metrics[i]}_{metrics[j]}")

# Generate heatmap
def plot_heatmap(results, metric, filename):
    data = np.array([[results[loss][model][metric] for loss in loss_functions] for model in models])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, xticklabels=loss_functions, yticklabels=models, cmap="coolwarm", linewidths=0.5)
    ax.set_title(f"Heatmap of {metric.upper()}")

    plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))    
    plt.savefig(f"./optuna_results/plots/heatmap_{metric}.png", bbox_inches="tight")
    plt.close()

# Generate heatmaps for each metric
for metric in metrics:
    plot_heatmap(results, metric, f"heatmap_{metric}")

print("All plots have been generated and saved.")