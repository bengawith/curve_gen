import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from functools import lru_cache
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from typing import Dict, List, Optional, Tuple


ROOT_DIR: str = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)

from src.utils_ import (
    set_seed,
    preprocess_from_dat,
    preprocess_kulfan_parameters,
    load_json,
    load_scalers,
)
from src.models_ import instantiate_model

set_seed(42)

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class CurveGenError(Exception):
    """Custom exception for CurveGen errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class CurveGen:
    __slots__ = (
        "k_params", "dat_path", "coords", "aero_name", "y_pred", "model", "params", "device",
        "X_scaler", "y_scaler", "model_path", "params_path", "scaler_dir", "dat_dir", "_prediction_cache",
        "_cache_size", "root_dir", "model_dir"
    )

    def __init__(self, k_params: Optional[Dict] = None, dat_path: Optional[str] = None, coords: Optional[List] = None, aero_name: Optional[str] = None, cache_size: int = 128, model_dir: Optional[str] = None):
        """
        Initializes the model and preprocesses the input data.
        Args:
            k_params: Kulfan parameters dict.
            dat_path: Path to .dat file.
            coords: List of coordinates.
            aero_name: Name of the aerofoil.
            cache_size: LRU cache size for predictions.
            model_dir: Directory containing model and scaler files.
        """
        self._validate_inputs(k_params, dat_path, coords, aero_name)  # Upgrade: Add validation
        self.k_params = k_params
        self.dat_path = dat_path
        self.coords = coords
        self.aero_name = aero_name
        self.y_pred = None
        self._cache_size = cache_size

        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = Path(model_dir) if model_dir else self.root_dir / 'curve_gen'
        self._setup_paths()
        self._initialize_model_and_scalers()
        self._setup_cache()
        self.y_pred = self._generate()

    def _validate_inputs(self, k_params, dat_path, coords, aero_name):  # Upgrade: New method for validation
        """Validate that exactly one input type is provided."""
        inputs = [k_params, dat_path, coords, aero_name]
        provided = [i for i in inputs if i is not None]
        if len(provided) != 1:
            raise CurveGenError("Exactly one input (k_params, dat_path, coords, or aero_name) must be provided.")

    def _setup_paths(self) -> None:
        """Set up all necessary file paths and verify existence."""
        self.model_path = self.model_dir / 'cg_model.pt'
        self.params_path = self.model_dir / 'cg_params.json'
        self.scaler_dir = self.model_dir / 'cg_scalers'
        self.dat_dir = self.root_dir / 'data/aerofoil_data'
        if not self.model_path.is_file():
            raise CurveGenError(f"Model file not found at {self.model_path}")
        if not self.params_path.is_file():
            raise CurveGenError(f"Parameters file not found at {self.params_path}")
            

    def _initialize_model_and_scalers(self) -> None:
        """Initialize the neural network model and scalers in one pass."""
        try:
            self.params = load_json(str(self.params_path))
            self.model = instantiate_model(
                net_name=self.params['net_name'],
                input_size=self.params['input_size'],
                output_size=self.params['output_size'],
                params=self.params
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.model.to(self.device)
            self.X_scaler, self.y_scaler = load_scalers(str(self.scaler_dir))
        except Exception as e:
            raise CurveGenError(f"Error initializing model or scalers: {str(e)}")

    def _setup_cache(self) -> None:
        """Set up LRU cache for predictions using functools.lru_cache."""  # Upgrade: Use built-in LRU
        self._prediction_cache = lru_cache(maxsize=self._cache_size)(self._cached_generate)

    def _cached_generate(self, input_key: str) -> np.ndarray:  # Upgrade: Refactored for caching
        """Cached version of prediction generation."""
        X_tensor = self._preprocess_input(input_key)
        if X_tensor.ndim == 2:
            X_tensor = X_tensor.unsqueeze(0)
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()
        return self.y_scaler.inverse_transform(y_pred_scaled).flatten()

    def _extract_params_from_key(self, input_key: str) -> Tuple[Optional[Dict], Optional[str], Optional[List]]:  # Upgrade: New helper
        """Extract k_params, aero_name, coords from input_key."""
        params = json.loads(input_key)
        if 'dat_path' in params:
            return preprocess_from_dat(file_path=params['dat_path'], dat_directory=str(self.dat_dir))
        elif 'aero_name' in params:
            return preprocess_from_dat(aerofoil_name=params['aero_name'], dat_directory=str(self.dat_dir))
        elif 'coords' in params:
            k_params = get_kulfan_parameters(coordinates=np.array(params['coords']), n_weights_per_side=6)
            return k_params, None, params['coords']
        else:
            return params['k_params'], None, None

    def _preprocess_input(self, input_key: str) -> torch.Tensor:
        """
        Preprocess input data for the model.
        Args:
            input_key: String representation of input parameters for caching.
        Returns:
            Preprocessed input tensor (torch.Tensor).
        """
        k_params, aero_name, coords = self._extract_params_from_key(input_key)  # Upgrade: Refactored
        # Update attributes only if not set
        if self.k_params is None:
            self.k_params = k_params
        if self.aero_name is None and aero_name is not None:
            self.aero_name = aero_name
        if self.coords is None and coords is not None:
            self.coords = coords
            
        if k_params is None:
            raise CurveGenError("No k_params available for preprocessing.")
            
        X_input = np.array(preprocess_kulfan_parameters(k_params), dtype=np.float32).reshape(1, -1)
        X_scaled = self.X_scaler.transform(X_input)
        return torch.from_numpy(X_scaled).float().to(self.device)

    def _generate(self) -> np.ndarray:
        """
        Processes the input, scales it, runs the model, and inverse transforms the predictions.
        Uses LRU cache for repeated predictions.
        Returns:
            y_pred: np.ndarray of predictions.
        """
        # Create input_key - convert numpy arrays to lists for JSON serialization
        if self.dat_path is not None:
            input_key = json.dumps({"dat_path": self.dat_path}, sort_keys=True)
        elif self.aero_name is not None:
            input_key = json.dumps({"aero_name": self.aero_name}, sort_keys=True)
        elif self.coords is not None:
            # Convert numpy arrays to lists for JSON serialization
            coords_serializable = self.coords if isinstance(self.coords, list) else self.coords.tolist()
            input_key = json.dumps({"coords": coords_serializable}, sort_keys=True)
        else:
            # Convert k_params dict values from numpy arrays to lists for JSON serialization
            if self.k_params is None:
                raise CurveGenError("No valid input provided for prediction generation.")
            k_params_serializable = {}
            for key, value in self.k_params.items():
                if isinstance(value, np.ndarray):
                    k_params_serializable[key] = value.tolist()
                else:
                    k_params_serializable[key] = value
            input_key = json.dumps({"k_params": k_params_serializable}, sort_keys=True)

        return self._prediction_cache(input_key)  # Upgrade: Use cached method

    def get_data(self) -> Dict:
        """
        Returns a dictionary with the prediction results.
        Returns:
            dict: {aero_name, k_params, predictions}
        """
        return {
            'aero_name': self.aero_name,
            'k_params': self.k_params,
            'predictions': self.y_pred,
        }

    def plot_curve(self, save_path: Optional[str] = None, **kwargs) -> None:  # Upgrade: Add save option and kwargs
        """
        Plots the lift coefficient (CL) curve for the given aerofoil.
        Args:
            save_path: Optional path to save the plot.
            **kwargs: Additional Matplotlib plot options.
        """
        if self.y_pred is None:
            raise CurveGenError("No predictions available. Ensure `_generate()` has been executed.")
        y_pred = self.y_pred
        n = len(y_pred) // 2
        cls = y_pred[:n]
        alphas = y_pred[n:]
        idx = np.argsort(alphas)
        alphas_sorted = np.array(alphas)[idx]
        cls_sorted = np.array(cls)[idx]
        plt.figure(figsize=(8, 6))
        plt.plot(alphas_sorted, cls_sorted, '-', label=(self.aero_name.upper() if self.aero_name else "Predicted Curve"), **kwargs)
        plt.xlabel("Angle of Attack (Alpha)")
        plt.ylabel("Lift Coefficient (CL)")
        plt.legend()
        plt.grid(True)
        plt.title(f"Predicted Lift Curve for {self.aero_name.upper() if self.aero_name else 'Aerofoil'}")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_aero(self, save_path: Optional[str] = None, **kwargs) -> None:  # Upgrade: Add save option and kwargs
        """
        Plots the aerofoil shape.
        Args:
            save_path: Optional path to save the plot.
            **kwargs: Additional Matplotlib plot options.
        """
        if self.coords is None:
            raise CurveGenError("No coordinates available. Ensure `_preprocess()` has been executed or provide coordinates.")
        coords = np.array(self.coords)
        x = coords[:, 0]
        y = coords[:, 1]
        plt.style.use('bmh')
        plt.figure(figsize=(8, 6))
        plt.ylim((-0.5, 0.5))
        plt.plot(x, y, label=(self.aero_name.upper() if self.aero_name else "Aerofoil"), **kwargs)
        plt.title(self.aero_name.upper() if self.aero_name else "Aerofoil")
        plt.legend(fontsize='small')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def save_data(self, file_path: str) -> None:  # Upgrade: New method to save data
        """Save prediction data to a JSON file."""
        data = self.get_data()
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
