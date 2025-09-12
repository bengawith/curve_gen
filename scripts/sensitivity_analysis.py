"""
Sensitivity Analysis for Aerodynamic Model

This script performs sensitivity analysis on a trained GRU model to understand
the impact of CST parameters on aerodynamic performance predictions.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import torch
from typing import Any, Tuple, Dict, List

# Add project root to Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)))))

from src.torch_models import instantiate_model
from src.torch_utils import set_seed, load_json, load_scalers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SensitivityAnalyzer:
    def __init__(self, model_dir: str, data_path: str):
        """
        Initialize the sensitivity analyzer with paths to model and data.
        
        Args:
            model_dir (str): Directory containing model files
            data_path (str): Path to the dataset CSV file
        """
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        
        # Model artifacts paths
        self.model_path = self.model_dir / 'models/gru_log.pt'
        self.params_path = self.model_dir / 'cg_params.json'
        self.scaler_path = self.model_dir / 'cg_scalers/'
        
        # Initialize components as None
        self.model = None
        self.x_scaler = None
        self.y_scaler = None
        self.params: Dict[str, Any] | None = None
        self.df: pd.DataFrame | None = None

    def setup(self) -> None:
        """Set up the analyzer by loading model, scalers, and data."""
        try:
            logger.info("Loading model parameters...")
            self.params: Dict[str, Any] = load_json(str(self.params_path))
            
            logger.info("Initializing model...")
            self.model = instantiate_model(
                net_name=self.params['net_name'],
                input_size=self.params['input_size'],
                output_size=self.params['output_size'],
                params=self.params
            )
            
            logger.info("Loading model state...")
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            
            logger.info("Loading scalers...")
            self.x_scaler, self.y_scaler = load_scalers(str(self.scaler_path))
            
            logger.info("Loading dataset...")
            self.df: pd.DataFrame = pd.read_csv(self.data_path)
            
        except Exception as e:
            logger.error(f"Error during setup: {str(e)}")
            raise

    def get_baseline_data(self, aerofoil_name: str = 'naca0024') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get baseline parameters for the specified aerofoil.
        
        Args:
            aerofoil_name (str): Name of the baseline aerofoil
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: CST parameters and other parameters
        """
        try:
            baseline_row = self.df[self.df['aerofoil_name'] == aerofoil_name].iloc[0]
            
            # Get CST parameters
            cst_cols = [f"lower_weight_{i}" for i in range(6)] + [f"upper_weight_{i}" for i in range(6)]
            baseline_params = baseline_row[cst_cols].values.astype(np.float32)
            
            # Get other parameters (TE, LE)
            other_params = baseline_row[["TE_thickness", "leading_edge_weight"]].values.astype(np.float32)
            
            return baseline_params, other_params
            
        except IndexError:
            logger.error(f"Aerofoil {aerofoil_name} not found in dataset")
            raise
        except Exception as e:
            logger.error(f"Error getting baseline data: {str(e)}")
            raise

    def predict_lift_curve(self, cst_params: np.ndarray, other_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict lift curve for given parameters.
        
        Args:
            cst_params (np.ndarray): CST parameters
            other_params (np.ndarray): Other parameters (TE, LE)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted alpha and CL values
        """
        try:
            # Combine and reshape parameters
            full_params = np.concatenate([cst_params.ravel(), other_params.ravel()])
            scaled_params = self.x_scaler.transform(full_params.reshape(1, -1))
            
            # Make prediction
            with torch.no_grad():
                pred_scaled = self.model(torch.tensor(scaled_params, dtype=torch.float32)).numpy()
            
            # Process prediction
            pred = self.y_scaler.inverse_transform(pred_scaled).flatten()
            cl = pred[:48]  # First 48 values are CL
            alpha = pred[48:]  # Remaining values are alpha
            
            return alpha, cl
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise

    def analyze_sensitivity(self, perturbation_factor: float = 0.15) -> Dict:
        """
        Perform sensitivity analysis by perturbing each CST parameter.
        
        Args:
            perturbation_factor (float): Fraction by which to perturb parameters
            
        Returns:
            Dict: Dictionary containing analysis results
        """
        try:
            # Get baseline data
            baseline_cst, baseline_other = self.get_baseline_data()
            baseline_alpha, baseline_cl = self.predict_lift_curve(baseline_cst, baseline_other)
            
            results = []
            curves = {'baseline': (baseline_alpha, baseline_cl)}
            
            # Analyze each CST parameter
            cst_cols = [f"lower_weight_{i}" for i in range(6)] + [f"upper_weight_{i}" for i in range(6)]
            
            for i, param_name in enumerate(cst_cols):
                # Format name for plot
                param_name = " ".join([x.capitalize() if x.isalpha() else str(int(x) + 1) for x in param_name.split('_')])

                # Create perturbed parameters
                perturbed_cst = baseline_cst.copy()
                perturbation = baseline_cst[i] * perturbation_factor
                perturbed_cst[i] += perturbation
                
                # Get predictions for perturbed parameters
                alpha, cl = self.predict_lift_curve(perturbed_cst, baseline_other)
                curves[param_name] = (alpha, cl)
                
                # Calculate sensitivity metrics
                cl_max_change = cl.max() - baseline_cl.max()
                results.append({
                    'Parameter': param_name,
                    'CL_max_Change': cl_max_change,
                    'Relative_Change': cl_max_change / baseline_cl.max()
                })
            
            return {
                'results': pd.DataFrame(results),
                'curves': curves,
                'baseline_data': (baseline_alpha, baseline_cl)
            }
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            raise

    def plot_results(self, analysis_results: Dict, save_dir: str = '.') -> None:
        """
        Generate and save plots from sensitivity analysis results.
        
        Args:
            analysis_results (Dict): Results from analyze_sensitivity
            save_dir (str): Directory to save plots
        """
        try:
            save_dir: Path = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Plot 1: Lift Curves
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot baseline
            baseline_alpha, baseline_cl = analysis_results['baseline_data']
            ax.plot(baseline_alpha, baseline_cl, 'k-', label='Baseline (NACA 0024)', linewidth=2)
            
            # Plot perturbed curves
            curves = analysis_results['curves']
            for param_name, (alpha, cl) in curves.items():
                if param_name != 'baseline':
                    ax.plot(alpha, cl, '--', label=f'Perturbed {param_name}', alpha=0.7)
            
            ax.set_xlabel('Angle of Attack (degrees)')
            ax.set_ylabel('Lift Coefficient (CL)')
            ax.set_title('Sensitivity of Lift Curve to CST Parameter Perturbations')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(save_dir / 'sensitivity_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Sensitivity Barplot
            fig, ax = plt.subplots(figsize=(12, 8))
            results_df = analysis_results['results']
            
            # Create color map based on values
            values = results_df['CL_max_Change'].values
            max_abs_val = max(abs(values))
            
            # Create a custom diverging colormap
            norm = mcolors.TwoSlopeNorm(vmin=-max_abs_val, vcenter=0, vmax=max_abs_val)
            cmap = cm.get_cmap('RdBu_r')  # Red-Blue diverging colormap, reversed so red is positive
            
            # Get colors for each value
            bar_colors = [cmap(norm(value)) for value in values]
            
            # Create bar plot with custom colors
            bars = ax.barh(
                y=range(len(results_df)),
                width=values,
                color=bar_colors
            )
            
            # Set y-axis labels
            ax.set_yticks(range(len(results_df)))
            ax.set_yticklabels(results_df['Parameter'])
            
            ax.axvline(0, color='black', linestyle='--', alpha=0.5)
            ax.set_title('Change in Maximum Lift Coefficient per Parameter', pad=20)
            ax.set_xlabel('Change in CL_max')
            
            # Add value labels with better positioning
            for i, v in enumerate(results_df['CL_max_Change']):
                # For negative values, place text to the left of the bar
                # For positive values, place text to the right of the bar
                if v < 0:
                    x_pos = v - (max_abs_val * 0.02)  # Small offset to the left
                    ha = 'right'
                else:
                    x_pos = v + (max_abs_val * 0.02)  # Small offset to the right
                    ha = 'left'
                
                ax.text(x_pos, i, f'{v:.4f}', 
                       va='center',
                       ha=ha,
                       fontsize=11,
                       bbox=dict(facecolor='white', 
                                edgecolor='none',
                                alpha=0.7,
                                pad=1))
            
            # Adjust plot margins to ensure labels are visible
            plt.margins(x=0.2)  # Add 20% padding on x-axis
            plt.tight_layout()
            plt.savefig(save_dir / 'sensitivity_barplot.png', dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in plotting: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize paths
        model_dir = r'C:\Users\Ben\workspaces\github.com\bengawith\project\curve_gen'
        data_path = r'C:\Users\Ben\workspaces\github.com\bengawith\project\data\csv\14KP_48CLA.csv'
        
        # Create and set up analyzer
        analyzer = SensitivityAnalyzer(model_dir, data_path)
        analyzer.setup()
        
        # Perform analysis
        results = analyzer.analyze_sensitivity(perturbation_factor=0.20)
        
        # Plot and save results
        save_dir = r'C:\Users\Ben\workspaces\github.com\bengawith\project\paper\figures'
        analyzer.plot_results(results, save_dir)
        
        logger.info("Sensitivity analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()
