import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
from functools import lru_cache
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters



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

from typing import Dict, List, Optional

set_seed(42)

class CurveGen:
    def __init__(self, k_params: Optional[Dict] = None, dat_path: Optional[str] = None, coords: Optional[List] = None, aero_name: Optional[str] = None, cache_size: int = 128, model_dir: Optional[str] = None):
        """Initializes the model and preprocesses the input data."""
        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_dir = Path(model_dir) if model_dir else self.root_dir / 'curve_gen'

        # Store input parameters
        self.k_params = k_params
        self.dat_path = dat_path
        self.coords = coords
        self.aero_name = aero_name
        self.y_pred = None

        # Initialise caching
        self.cache_size = cache_size

        # Set up paths
        self._setup_paths()
        
        # Initialize model and scalers
        self._initialize_model()
        self._load_scalers()
        
        # Cache for computations
        self._setup_cache(cache_size)

        # Generate the predictions
        self.y_pred = self._generate()

    def _setup_paths(self) -> None:
        """Set up all necessary file paths."""
        self.model_path = self.model_dir / 'cg_model.pt'
        self.params_path = self.model_dir / 'cg_params.json'
        self.scaler_dir = self.model_dir / 'cg_scalers'
        self.dat_dir = self.root_dir / 'data/aerofoil_data'
        
        # Verify paths exist
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not self.params_path.exists():
            raise FileNotFoundError(f"Parameters file not found at {self.params_path}")
            
    def _initialize_model(self) -> None:
        """Initialize the neural network model."""
        try:
            self.params = load_json(str(self.params_path))
            # Load model
            self.model = instantiate_model(
                net_name=self.params['net_name'],
                input_size=self.params['input_size'],
                output_size=self.params['output_size'],
                params=self.params
            )
            self.model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device)))
            self.model.to(self.device)

        except Exception as e:
            raise RuntimeError(f"Error initializing model: {str(e)}")
        

    def _load_scalers(self) -> None:
        """Load and initialize data scalers."""
        try:
            self.X_scaler, self.y_scaler = load_scalers(str(self.scaler_dir))
        except Exception as e:
            raise RuntimeError(f"Error loading scalers: {str(e)}")


    def _setup_cache(self, cache_size: int) -> None:
        """Set up computation caching."""
        self._prediction_cache = {}
        self._cache_size = cache_size


    @lru_cache(maxsize=128)
    def _preprocess_input(self, input_key: str) -> torch.Tensor:
        """
        Preprocess input data with caching.
        
        Args:
            input_key: String representation of input parameters for caching.
            
        Returns:
            Preprocessed input tensor.
        """
        # Convert input key back to parameters
        params = json.loads(input_key)
        
        if 'dat_path' in params:
            self.k_params, self.aero_name, self.coords = preprocess_from_dat(file_path=params['dat_path'], dat_directory=str(self.dat_dir))
        elif 'aero_name' in params:
            self.k_params, self.aero_name, self.coords = preprocess_from_dat(aerofoil_name=params['aero_name'], dat_directory=str(self.dat_dir))
        elif 'coords' in params:
            self.k_params = get_kulfan_parameters(coordinates=np.array(params['coords']), n_weights_per_side=6)
        else:
            self.k_params = params['k_params']

        X_input = np.array(preprocess_kulfan_parameters(self.k_params)).reshape(1, -1)
        return torch.tensor(self.X_scaler.transform(X_input), dtype=torch.float32).to(self.device)


    def _generate(self):
        """Processes the input, scales it, runs the model, and inverse transforms the predictions."""
        
        if self.dat_path is not None:
            input_key: str = json.dumps({"dat_path": self.dat_path}, sort_keys=True)
        elif self.aero_name is not None:
            input_key: str = json.dumps({"aero_name": self.aero_name}, sort_keys=True)
        elif self.coords is not None:
            input_key: str = json.dumps({"coords": self.coords}, sort_keys=True)
        else:
            input_key: str = json.dumps({"k_params": self.k_params}, sort_keys=True)
        
        if input_key in self._prediction_cache:
            return self._prediction_cache[input_key]

        X_tensor: torch.Tensor = self._preprocess_input(input_key)

        # Ensure GRU model input shape is correct
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)

        # Generate model prediction
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        # Convert predictions back to original scale
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled).flatten()

        # Cache predictions
        self._prediction_cache[input_key] = y_pred
            
        # Manage cache size
        if len(self._prediction_cache) > self._cache_size:
            self._prediction_cache.pop(next(iter(self._prediction_cache)))
        
        return y_pred



    def get_data(self):
        """Returns a dictionary with the prediction results."""
        return {
            'aero_name': self.aero_name,
            'k_params': self.k_params,
            'predictions': self.y_pred,
        }


    def plot_curve(self):
        """Plots the lift coefficient (CL) curve for the given aerofoil."""
        if self.y_pred is None:
            raise ValueError("No predictions available. Ensure `_generate()` has been executed.")

        num_values = len(self.y_pred) // 2
        cls = self.y_pred[:num_values]  # First 48 values: CLs
        alphas = self.y_pred[num_values:]  # Last 48 values: Alphas

        sorted_indices = np.argsort(alphas)
        alphas_sorted = alphas[sorted_indices]
        cls_sorted = cls[sorted_indices]

        plt.figure(figsize=(8, 6))
        plt.plot(alphas_sorted, cls_sorted, '-', label=self.aero_name.upper() if self.aero_name else "Predicted Curve", linewidth=2)
        plt.xlabel("Angle of Attack (Alpha)")
        plt.ylabel("Lift Coefficient (CL)")
        plt.legend()
        plt.grid(True)
        plt.title(f"Predicted Lift Curve for {self.aero_name.upper() if self.aero_name else 'Aerofoil'}")
        plt.show()


    def plot_aero(self):
        """Plots the aerofoil shape."""
        if self.coords is None:
            raise ValueError("No coordinates available. Ensure `_preprocess()` has been executed or provide coordinates.")

        x = [x[0] for x in self.coords]
        y = [x[1] for x in self.coords]

        plt.style.use('bmh')
        plt.figure(figsize=(8, 6))
        plt.ylim((-0.5, 0.5))
        plt.plot(x, y, label=f'{self.aero_name.upper() if self.aero_name else "Aerofoil"}', linewidth=2)
        plt.title(f"{self.aero_name.upper()}" if self.aero_name else "Aerofoil")
        plt.legend(fontsize='small')
        plt.show()

