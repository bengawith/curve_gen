import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')

import numpy as np
import seaborn as sns
import shap
import os
import pandas as pd
import logging
from typing import Optional
import torch


logger = logging.getLogger(__name__)

CSV_PATH = './data/csv/14KP_48CLA.csv'

class CurveVisualiser:
    def __init__(self, predictions: np.ndarray, targets: np.ndarray):
        self.predictions = np.array(predictions)
        self.targets = np.array(targets)

    def plot_training_loss(self, train_losses: np.ndarray, val_losses: Optional[np.ndarray] = None, title: str = "Training Loss Curve", save_path: Optional[str] = None) -> None:
        plt.figure(figsize=(8, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label='Training Loss', linestyle='-')
        if val_losses is not None:
            plt.plot(epochs, val_losses, label='Validation Loss', linestyle='--')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Training loss curve saved to {save_path}")
        plt.show()

    def plot_variance(self, actual: np.ndarray, predicted: np.ndarray, title: str, x_label: str = "Actual Values", y_label: str = "Predicted Values", show: bool = True) -> None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual.flatten(), y=predicted.flatten(), color='orange', s=10, label='Predicted Values')
        sns.scatterplot(x=actual.flatten(), y=actual.flatten(), color='blue', s=10, label='Actual Values')
        #sns.regplot(x=actual.flatten(), y=predicted.flatten(), scatter=False, color="red", line_kws={"linewidth": 2})
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()


    def plot_variance_eval_table(self, actual: np.ndarray, predicted: np.ndarray, title: str, 
                        x_label: str = "Actual Values", y_label: str = "Predicted Values", 
                        show: bool = True) -> None:
        """
        Plots variance between actual and predicted values and displays a table below the plot.

        Args:
            actual (np.ndarray): Array of actual values.
            predicted (np.ndarray): Array of predicted values.
            title (str): Title of the plot.
            x_label (str, optional): Label for the X-axis. Defaults to "Actual Values".
            y_label (str, optional): Label for the Y-axis. Defaults to "Predicted Values".
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        fig, ax = plt.subplots(figsize=(10, 8))  # Increase figure size for better visualization

        # Scatter plot of actual vs predicted values
        sns.scatterplot(x=actual.flatten(), y=actual.flatten(), color='blue', s=10, label='Actual Values', ax=ax)
        sns.scatterplot(x=actual.flatten(), y=predicted.flatten(), color='orange', s=10, label='Predicted Values', ax=ax)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        # Ensure rounding before passing to table
        actual_rounded = np.round(actual.flatten(), 4)
        predicted_rounded = np.round(predicted.flatten(), 4)

        # Create formatted table data
        table_data = np.column_stack((actual_rounded, predicted_rounded)).tolist()

        colLabels = ['Actual', 'Predicted']

        # Create table below the plot
        table = plt.table(cellText=table_data,
                        colLabels=colLabels,
                        loc='bottom',
                        cellLoc='center',
                        bbox=[0.0, -0.4, 1.0, 0.3],  # Adjust position for better readability
                        cellColours=[['lightgrey'] * 2] * len(table_data),
                        colWidths=[.5]*2,
                        )
        cellDict = table.get_celld()
        for i in range(0,len(colLabels)):
            cellDict[(0,i)].set_height(10)
            for j in range(1,len(table_data)+1):
                cellDict[(j,i)].set_height(10)
        
        table.set_fontsize(10)

        # Adjust spacing to prevent table overlap
        plt.subplots_adjust(bottom=0.4)  

        if show:
            plt.show()


    def plot_variance_eval(self, actual: np.ndarray, predicted: np.ndarray, title: str, x_label: str = "Actual Values", y_label: str = "Predicted Values", show: bool = True) -> None:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual.flatten(), y=actual.flatten(), color='blue', s=10, label='Actual Values')
        sns.scatterplot(x=actual.flatten(), y=predicted.flatten(), color='orange', s=10, label='Predicted Values')
        #sns.regplot(x=actual.flatten(), y=predicted.flatten(), scatter=False, color="red", line_kws={"linewidth": 2})
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if show:
            plt.show()

    def plot_actual_vs_predicted_eval(self) -> None:
        if self.predictions.ndim == 1:
            self.predictions = self.predictions.reshape(1, -1)
        if self.targets.ndim == 1:
            self.targets = self.targets.reshape(1, -1)
        
        self.plot_variance_eval(self.targets[:, :48], self.predictions[:, :48], "Actual vs Predicted CL")
        self.plot_variance_eval(self.targets[:, -48:], self.predictions[:, -48:], "Actual vs Predicted Alpha")


    def plot_actual_vs_predicted(self) -> None:
        if self.predictions.ndim == 1:
            self.predictions = self.predictions.reshape(1, -1)
        if self.targets.ndim == 1:
            self.targets = self.targets.reshape(1, -1)
        
        self.plot_variance(self.targets[:, :48], self.predictions[:, :48], "Actual vs Predicted CL")
        self.plot_variance(self.targets[:, -48:], self.predictions[:, -48:], "Actual vs Predicted Alpha")


    def plot_comparison(self, num_samples_to_plot: int = 5) -> None:
        colours = plt.cm.viridis(np.linspace(0, 1, num_samples_to_plot))
        for i in range(num_samples_to_plot):
            y_pred_alphas = self.predictions[i][-48:]
            y_test_alphas = self.targets[i][-48:]
            y_pred_cls = self.predictions[i][:48]
            y_test_cls = self.targets[i][:48]
            plt.plot(y_test_alphas, y_test_cls, '-', color=colours[i], alpha=0.85, label=f'Sample {i + 1} Actual CL')
            plt.plot(y_pred_alphas, y_pred_cls, '--', color=colours[i], alpha=0.85, label=f'Sample {i + 1} Predicted CL')
        plt.xlabel('Angle of Attack')
        plt.ylabel('CL Values')
        plt.title('Comparison of Predicted and Actual CL Values')
        plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def single_plot(self, idx: int, aero_name: Optional[str] = None, metrics=None) -> None:
        colours = plt.cm.viridis(np.linspace(0, 1, 4))
        y_pred_alphas = self.predictions[idx][-48:]
        y_test_alphas = self.targets[idx][-48:]
        y_pred_cls = self.predictions[idx][:48]
        y_test_cls = self.targets[idx][:48]
        plt.plot(y_test_alphas, y_test_cls, '-', color=colours[1], alpha=0.85, label=f'{aero_name or idx} Actual CL')
        plt.plot(y_pred_alphas, y_pred_cls, '--', color=colours[1], alpha=0.85, label=f'{aero_name or idx} Predicted CL')
        if metrics is not None:
            plt.text(0.05, 0.95, f'MAE: {metrics["mae"]:.4f}\nRMSE: {metrics["rmse"]:.4f}\nMSE: {metrics["mse"]:.4f}\nR2: {metrics["r2"]:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.xlabel('Angle of Attack')
        plt.ylabel('CL Values')
        plt.title('Comparison of Predicted and Actual CL Values')
        plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def single_plot_CurveGen(self, aero_name: Optional[str] = None, metrics=None) -> None:
        colours = plt.cm.viridis(np.linspace(0, 1, 4))
        y_pred_alphas = self.predictions[-48:]
        y_test_alphas = self.targets[-48:]
        y_pred_cls = self.predictions[:48]
        y_test_cls = self.targets[:48]
        plt.plot(y_test_alphas, y_test_cls, '-', color=colours[2], label=f'{aero_name.upper() or ""} XFoil')
        plt.plot(y_pred_alphas, y_pred_cls, '--', color=colours[1], label=f'{aero_name.upper() or ""} Predicted Curve')
        if metrics is not None:
            plt.text(0.05, 0.95, f'MAE: {metrics["mae"]:.4f}\nRMSE: {metrics["rmse"]:.4f}\nMSE: {metrics["mse"]:.4f}\nR2: {metrics["r2"]:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        #plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xlabel('Angle of Attack')
        plt.ylabel('CL Values')
        plt.title('Comparison of Predicted Curve and XFoil Curve')
        plt.legend(fontsize='small', loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.show()

    def generate_shap_plots(self, model: torch.nn.Module, train_data: torch.Tensor, test_data: torch.Tensor, feature_names: list, output_labels: list, show: bool = True) -> None:
        explainer = shap.DeepExplainer(model, train_data)
        shap_values = explainer.shap_values(test_data)
        for sample_index, test_sample in enumerate(test_data):
            for output_index, label in enumerate(output_labels):
                shap.summary_plot(
                    shap_values[output_index][sample_index],
                    test_sample.cpu().numpy(),
                    feature_names=feature_names,
                    plot_type="bar",
                    show=show
                )
                plt.title(f"SHAP Plot for Sample {sample_index + 1}, Output: {label}")
                if show:
                    plt.show()

class TrainPlot:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.X_cols = []
        self.y_cols = []
        self.df = None
        self._load_data()

    def _load_data(self) -> None:
        if not os.path.exists(self.dataset):
            raise FileNotFoundError(f"CSV file not found: {self.dataset}")
        file = os.path.splitext(os.path.basename(self.dataset))[0]
        try:
            n_weights = int((int(file.split("KP")[0]) - 2) / 2)
            n_outs = int(file.split("_")[1].replace("CLA", ""))
        except (IndexError, ValueError):
            raise ValueError("Filename does not match expected format: '14KP_48CLA.csv'.")
        self.df = pd.read_csv(self.dataset)
        self.X_cols = [f"lower_weight_{i}" for i in range(n_weights)] + [f"upper_weight_{i}" for i in range(n_weights)] + ["TE_thickness", "leading_edge_weight"]
        self.y_cols = [f"CL_{i}" for i in range(n_outs)] + [f"alpha_{i}" for i in range(n_outs)]
        for col in self.X_cols + self.y_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing column in dataset: {col}")

    def visualize_features(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call _load_data() first.")
        for feature in self.X_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[feature], kde=True, bins=30, color="blue", alpha=0.7)
            plt.title(f"Distribution of {feature}")
            plt.xlabel(feature)
            plt.ylabel("Frequency")
            plt.show()

    def visualize_pairplots(self) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call _load_data() first.")
        sns.pairplot(self.df[self.X_cols], diag_kind="kde", corner=True)
        plt.suptitle("Pair Plot of Kulfan Parameters", y=1.02)
        plt.show()

    def plot_cl_vs_alpha(self, aerofoil_names: list) -> None:
        if self.df is None:
            raise RuntimeError("Data not loaded. Call _load_data() first.")
        plt.figure(figsize=(12, 6))
        for aerofoil in aerofoil_names:
            aerofoil_data = self.df[self.df['aerofoil_name'] == aerofoil]
            alpha_cols = [col for col in self.df.columns if "alpha_" in col]
            cl_cols = [col for col in self.df.columns if "CL_" in col]
            for i in range(len(alpha_cols)):
                plt.plot(aerofoil_data[alpha_cols[i]], aerofoil_data[cl_cols[i]], label=f"{aerofoil} (alpha_{i})")
        plt.title("CL vs Alpha for Selected Aerofoils")
        plt.xlabel("Alpha (Degrees)")
        plt.ylabel("CL (Lift Coefficient)")
        plt.legend()
        plt.show()