import os
import json
import logging
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from typing import Optional, Dict, Any, Tuple, List

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


env = os.getenv('.env')


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_time(start_time: float, end_time: float, ret=False) -> None:
    """Format runtime in H:M:S format and log it."""
    runtime = end_time - start_time
    if runtime >= 3600:
        hrs = int(runtime // 3600)
        mins = int((runtime % 3600) // 60)
        secs = runtime % 60
        runtime_str = f"Runtime: {runtime:.2f}s or {hrs:02d}:{mins:02d}:{secs:.2f}"
    elif runtime >= 60:
        mins = int(runtime // 60)
        secs = runtime % 60
        runtime_str = f"Runtime: {runtime:.2f}s or {mins:02d}:{secs:.2f}"
    else:
        runtime_str = f"Runtime: {runtime:.2f}s"
    logger.info(runtime_str)

    if ret:
        return runtime
    

def find_file(file: str = None, dir: str = None) -> str:
    """Search for a file in predefined directories."""
    search_root = Path("/" if os.name != "nt" else "C:\\Users")
    if file is not None and dir is None:
        for root, _, files in os.walk(search_root):
            if file in files:
                return str(Path(root) / file)
        raise FileNotFoundError(f"The file {file} was not found.")
    elif file is None and dir is not None:
        for root, dirs, _ in os.walk(search_root):
            if dir in dirs:
                return str(Path(root) / dir)
        raise FileNotFoundError(f"The directory {dir} was not found.")
    else:
        raise ValueError("Please enter a valid filename.")
    

def read_coords(file_path: str, n_digits: int = 6) -> np.array:
    """
    Read coordinates from a file.

    Parameters:
        file_path (str): Path to the file.
        n_digits (int): Number of digits to round to.

    Returns:
        np.array: Array of coordinates.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
        return np.array([[round(float(p), n_digits) for p in line.strip().split()] for line in lines if len(line.strip().split()) == 2])


def preprocess_from_dat(file_path: Optional[str]=None, aerofoil_name: Optional[str]=None, coords: Optional[list]=None, dat_directory: str='./data/aerofoil_data') -> Tuple[Dict[str, Any], str]:
    """
    Preprocess an aerofoil .dat file or aerofoil name into Kulfan parameters.

    Parameters:
        file_path (str): Path to the .dat file.
        aerofoil_name (str): Name of the aerofoil.
        coords (list): List of coordinates.
        dat_directory (str): Directory containing aerofoil .dat files.

    Returns:
        Tuple[Dict[str, Any], str]: Kulfan parameters and aerofoil name.
    """
    if file_path is None and aerofoil_name is not None:
        filename = aerofoil_name.lower() + ".dat"
        if dat_directory and filename in os.listdir(dat_directory):
            file_path = os.path.join(dat_directory, filename)
        else:
            raise FileNotFoundError(f"Aerofoil '{aerofoil_name}' not found in '{dat_directory}'.")
    if file_path and file_path.endswith(".dat"):
        try:
            coordinates = read_coords(file_path)
            kulfan_params = get_kulfan_parameters(coordinates=coordinates, n_weights_per_side=6)
            aerofoil_n = os.path.splitext(os.path.basename(file_path))[0]
            return kulfan_params, aerofoil_n, coordinates
        except Exception as e:
            raise ValueError(f"Error processing {file_path}: {e}")
    if coords is not None:
        try:
            coordinates = np.array(coords)
            kulfan_params = get_kulfan_parameters(coordinates=coordinates, n_weights_per_side=6)
            return kulfan_params
        except Exception as e:
            raise ValueError(f"Error processing coordinates: {e}")
    raise ValueError("Invalid input. Provide either a file path or an aerofoil name.")


def preprocess_kulfan_parameters(kulfan_params: Dict[str, Any], n_weights: int = 6) -> np.ndarray:
    """
    Convert Kulfan parameters into a structured input array for the model.
    """
    return np.hstack([
        kulfan_params["lower_weights"][:n_weights],
        kulfan_params["upper_weights"][:n_weights],
        kulfan_params["TE_thickness"],
        kulfan_params["leading_edge_weight"]
    ])


def save_scalers(X_scaler: StandardScaler, y_scaler: MinMaxScaler, output_path: str) -> None:
    """
    Save scalers to a file.
    """
    os.makedirs(output_path, exist_ok=True)
    joblib.dump(X_scaler, output_path + "X_scaler.gz")
    joblib.dump(y_scaler, output_path + "y_scaler.gz")
    logger.info(f"Scalers saved to {output_path}_X_scaler.gz and {output_path}_y_scaler.gz")


def load_scalers(input_path: str) -> Tuple[StandardScaler, MinMaxScaler]:
    """
    Load scalers from a file.
    """
    if not input_path.endswith("\\") and not input_path.endswith("/"):
        input_path += "/"
    X_scaler = joblib.load(input_path + "X_scaler.gz")
    y_scaler = joblib.load(input_path + "y_scaler.gz")
    return X_scaler, y_scaler


def load_json(json_path: str) -> Any:
    """
    Load a JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")
    return json.load(open(json_path))


def serialize_state_dict(state_dict: dict) -> dict:
    serializable = {}
    for key, tensor in state_dict.items():
        serializable[key] = tensor.cpu().detach().tolist()
    return serializable


def deserialize_state_dict(state_dict: dict) -> dict:
    deserializable = {}
    for key, value in state_dict.items():
        deserializable[key] = torch.tensor(value)
    return deserializable


def load_model_from_state_dict(model: torch.nn.Module, model_path: str, device: str = "cpu") -> torch.nn.Module:
    return model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(array, dtype=torch.float32)


def evaluate_aerofoil_predictions(csv_file, model, X_scaler, y_scaler, device = 'cuda' if torch.cuda.is_available() else 'cpu',
                                  compute_metrics=True, plot=True, plot_sample: Optional[list[str]] = None, variance=False):
    """
    Iterate over all aerofoils in the CSV, predict the CL and Alpha curves, 
    plot the prediction vs actual curves (with aerofoil name), and optionally 
    compute the MSE and R² score for each aerofoil.

    If plot_sample is provided (a list of aerofoil names), only those aerofoils will be plotted.
    If not provided, and if plot=True, every aerofoil is plotted.
    
    Args:
        csv_file (str): Path to the CSV file containing aerofoil data.
        model (torch.nn.Module): Trained PyTorch model.
        X_scaler (sklearn.preprocessing.StandardScaler): Fitted scaler for input features.
        y_scaler (sklearn.preprocessing.MinMaxScaler): Fitted scaler for target values.
        device (str): Device ("cpu" or "cuda") on which the model runs.
        compute_metrics (bool): If True, compute and print MSE and R² scores.
        plot (bool): If True, display a plot for each aerofoil.
        plot_sample (list, optional): List of aerofoil names to plot. If provided, only those samples will be plotted.
    
    Returns:
        dict: A dictionary mapping aerofoil names to a dictionary of metrics (MSE and R²).
    """
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Determine the feature and label columns based on naming conventions
    lower_weight_cols = sorted(
        [col for col in df.columns if col.startswith("lower_weight_")],
        key=lambda x: int(x.split("_")[-1])
    )
    upper_weight_cols = sorted(
        [col for col in df.columns if col.startswith("upper_weight_")],
        key=lambda x: int(x.split("_")[-1])
    )
    CL_cols = sorted(
        [col for col in df.columns if col.startswith("CL_")],
        key=lambda x: int(x.split("_")[-1])
    )
    alpha_cols = sorted(
        [col for col in df.columns if col.startswith("alpha_")],
        key=lambda x: int(x.split("_")[-1])
    )
    
    # Check if the aerofoil name column exists; if not, create a default name.
    has_name = "aerofoil_name" in df.columns
    
    metrics_dict = {}

    #save_scalers(X_scaler, y_scaler, "./final_model/scalers/")

    # Iterate over each row (each aerofoil)
    for idx, row in df.iterrows():
        aerofoil_name = row["aerofoil_name"] if has_name else f"Sample_{idx}"
        
        # Build input vector in the same order as in training:
        # [lower_weights, upper_weights, TE_thickness, leading_edge_weight]
        lower_weights = [row[col] for col in lower_weight_cols]
        upper_weights = [row[col] for col in upper_weight_cols]
        TE_thickness = row["TE_thickness"]
        leading_edge_weight = row["leading_edge_weight"]
        X_input = np.hstack([lower_weights, upper_weights, [TE_thickness], [leading_edge_weight]])

        # Actual target outputs
        actual_cls = np.array([row[col] for col in CL_cols])
        actual_alphas = np.array([row[col] for col in alpha_cols])
        
        # Scale the input using the scaler fitted during training
        X_scaled = X_scaler.transform([X_input])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        # Generate model predictions
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy()
        # Convert scaled predictions back to original scale
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        predicted_cls = y_pred[0][:len(CL_cols)]
        predicted_alphas = y_pred[0][len(CL_cols):]
        
        # Combine predictions and actual values for metric computation and plotting
        predictions = np.concatenate([predicted_cls, predicted_alphas])
        targets = np.concatenate([actual_cls, actual_alphas])
        
        # Compute metrics if requested
        mse_val = None
        r2_val = None
        mae_val = None
        rmse_val = None
        if compute_metrics:
            mse_val = mean_squared_error(targets, predictions)
            mae_val = mean_absolute_error(targets, predictions)
            r2_val = r2_score(targets, predictions)
            rmse_val = np.sqrt(mse_val)
            metrics_dict[aerofoil_name] = {"predictions": predictions.tolist(), "targets": targets.tolist(), "mse": mse_val, "rmse": rmse_val, 'mae': mae_val, "r2": r2_val}

        if plot and ((plot_sample is None) or (aerofoil_name.lower() in [x.lower() for x in plot_sample])):
            from src.plot_ import CurveVisualiser
            visualiser = CurveVisualiser(predictions=predictions, targets=targets)
            visualiser.single_plot_CurveGen(aerofoil_name, metrics=metrics_dict[aerofoil_name])
            if variance:    
                visualiser.plot_actual_vs_predicted_eval()
            

            
        if compute_metrics and not plot:    
            print(f"Aerofoil: {aerofoil_name}")
            print(f"  MSE: {mse_val:.4f}, R²: {r2_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
    
    return metrics_dict


def eval_aero_preds(data_source, model, X_scaler, y_scaler, 
                                  device='cuda' if torch.cuda.is_available() else 'cpu',
                                  compute_metrics=True, plot=True, 
                                  prep_method="standard", plot_sample: Optional[List[str]] = None):
    """
    Evaluate each aerofoil’s predicted lift curve and compute metrics.
    
    This function supports three preparation methods:
      - "standard": Read the CSV directly and build the input vector from the standard feature columns.
      - "FeatureBinning": Data was prepared using the FeatureBinning class.
      - "DynamicBinning": Data was prepared using the DynamicBinning class.
      
    In the "standard" case, the CSV is read and processed row by row. For the binning methods, the corresponding
    DataPrep class is instantiated (which must store the full DataFrame in self.df and the scaled inputs/targets in
    self.X and self.y). Then, for each row, the appropriate input (from the scaled feature matrix) and target (from
    the scaled target array) are used to predict and compute metrics.
    
    If plot_sample is provided (a list of aerofoil names), only those aerofoils will be plotted.
    If not provided, and if plot=True, every aerofoil is plotted.
    
    Assumptions:
      - The feature columns are:
            lower_weight_0, ..., lower_weight_{n_weights-1}, upper_weight_0, ..., upper_weight_{n_weights-1},
            TE_thickness, leading_edge_weight.
      - The target columns are:
            CL_0, ..., CL_47, alpha_0, ..., alpha_47.
        (i.e. first the CL values then the alpha values)
      - The model output is a flat 96-dimensional vector that is inverse-transformed and then split into two halves:
            first 48 for CL, last 48 for alpha.
    
    Args:
      data_source (str): Path to the CSV file.
      model (torch.nn.Module): Trained model.
      X_scaler: Scaler for inputs.
      y_scaler: Scaler for targets.
      device (str): 'cpu' or 'cuda'.
      compute_metrics (bool): If True, compute and store metrics.
      plot (bool): If True, plot the lift curve for each aerofoil (subject to plot_sample).
      prep_method (str): One of "standard", "FeatureBinning", or "DynamicBinning".
      plot_sample (list, optional): List of aerofoil names to plot. If provided, only those samples will be plotted.
    
    Returns:
      dict: For "standard": mapping aerofoil name -> metrics; 
            for the binning methods: overall metrics computed per aerofoil.
    """
    metrics_dict = {}
    
    if prep_method.lower() == "standard":
        df = pd.read_csv(data_source)
        
        lower_weight_cols = sorted(
            [col for col in df.columns if col.startswith("lower_weight_")],
            key=lambda x: int(x.split("_")[-1])
        )
        upper_weight_cols = sorted(
            [col for col in df.columns if col.startswith("upper_weight_")],
            key=lambda x: int(x.split("_")[-1])
        )
        CL_cols = sorted(
            [col for col in df.columns if col.startswith("CL_")],
            key=lambda x: int(x.split("_")[-1])
        )
        alpha_cols = sorted(
            [col for col in df.columns if col.startswith("alpha_")],
            key=lambda x: int(x.split("_")[-1])
        )
        
        has_name = "aerofoil_name" in df.columns
        
        for idx, row in df.iterrows():
            aerofoil_name = row["aerofoil_name"] if has_name else f"Sample_{idx}"
            # Build input vector:
            lower_weights = [row[col] for col in lower_weight_cols]
            upper_weights = [row[col] for col in upper_weight_cols]
            TE_thickness = row["TE_thickness"]
            leading_edge_weight = row["leading_edge_weight"]
            X_input = np.hstack([lower_weights, upper_weights, [TE_thickness], [leading_edge_weight]])
            
            # Targets:
            actual_cls = np.array([row[col] for col in CL_cols])
            actual_alphas = np.array([row[col] for col in alpha_cols])
            
            X_scaled = X_scaler.transform([X_input])
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                y_pred_scaled = model(X_tensor).cpu().numpy()  # shape (1,96)
            y_pred = y_scaler.inverse_transform(y_pred_scaled)
            predicted_cls = y_pred[0][:len(CL_cols)]
            predicted_alphas = y_pred[0][len(CL_cols):]
            
            # Combine predictions and targets
            predictions = np.concatenate([predicted_cls, predicted_alphas])
            targets = np.concatenate([actual_cls, actual_alphas])
            
            if compute_metrics:
                mse_val = mean_squared_error(targets, predictions)
                mae_val = mean_absolute_error(targets, predictions)
                r2_val = r2_score(targets, predictions)
                rmse_val = np.sqrt(mse_val)
                metrics_dict[aerofoil_name] = {"mse": mse_val, "rmse": rmse_val, "mae": mae_val, "r2": r2_val}
            # Plot only if plot_sample is None or aerofoil_name is in plot_sample
            if plot and ((plot_sample is None) or (aerofoil_name.lower() in [x.lower() for x in plot_sample])):
                from src.plot_ import CurveVisualiser
                visualiser = CurveVisualiser(predictions=predictions, targets=targets)
                visualiser.single_plot_CurveGen(aerofoil_name, metrics=metrics_dict.get(aerofoil_name))
            elif compute_metrics and not plot:
                print(f"Aerofoil: {aerofoil_name}")
                print(f"  MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
                
        return metrics_dict
    
    elif prep_method in ["featurebinning", "dynamicbinning", "FeatureBinning", "DynamicBinning"]:
        # Use corresponding DataPrep class. For exact strings, we expect:
        # "FeatureBinning" or "DynamicBinning"
        if prep_method.lower() == "featurebinning":
            from src.data_ import DataPrepFeatureBinning as DPClass
        else:  # dynamicbinning
            from src.data_ import DataPrepDynamicBinning as DPClass
        dp_instance = DPClass(data_source)
        df = dp_instance.df  # The full DataFrame must be stored in self.df
        local_y_scaler = dp_instance.y_scaler
        
        for idx, row in df.iterrows():
            aerofoil_name = row["aerofoil_name"] if "aerofoil_name" in df.columns else f"Sample_{idx}"
            
            # Build input vector from original feature columns
            n_weights = 6  # adjust if necessary
            lower_weights = [row[f"lower_weight_{i}"] for i in range(n_weights)]
            upper_weights = [row[f"upper_weight_{i}"] for i in range(n_weights)]
            TE_thickness = row["TE_thickness"]
            leading_edge_weight = row["leading_edge_weight"]
            X_input = np.hstack([lower_weights, upper_weights, [TE_thickness], [leading_edge_weight]])
            
            # Targets:
            actual_cls = np.array([row[f"CL_{i}"] for i in range(48)])
            actual_alphas = np.array([row[f"alpha_{i}"] for i in range(48)])
            
            # Instead of re-scaling, use the already computed dp_instance.X (scaled) 
            # and dp_instance.y (scaled). Here, we use the row index:
            X_scaled = dp_instance.X[idx, :].reshape(1, -1)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                y_pred_scaled = model(X_tensor).cpu().numpy()  # shape (1,96)
            y_pred = local_y_scaler.inverse_transform(y_pred_scaled)
            predicted_cls = y_pred[0][:48]
            predicted_alphas = y_pred[0][48:]
            
            predictions = np.concatenate([predicted_cls, predicted_alphas])
            targets = np.concatenate([actual_cls, actual_alphas])
            
            if compute_metrics:
                mse_val = mean_squared_error(targets, predictions)
                mae_val = mean_absolute_error(targets, predictions)
                r2_val = r2_score(targets, predictions)
                rmse_val = np.sqrt(mse_val)
                metrics_dict[aerofoil_name] = {"mse": mse_val, "rmse": rmse_val, "mae": mae_val, "r2": r2_val}
            if plot and ((plot_sample is None) or (aerofoil_name.lower() in [x.lower() for x in plot_sample])):
                from src.plot_ import CurveVisualiser
                visualiser = CurveVisualiser(predictions=predictions, targets=targets)
                visualiser.single_plot_CurveGen(aerofoil_name, metrics=metrics_dict.get(aerofoil_name))
            elif compute_metrics and not plot:
                print(f"Aerofoil: {aerofoil_name}")
                print(f"  MSE: {mse_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}")
        return metrics_dict
    else:
        raise ValueError("prep_method must be 'standard', 'FeatureBinning', or 'DynamicBinning'")