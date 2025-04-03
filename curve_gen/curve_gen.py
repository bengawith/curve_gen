import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils_ import (
    set_seed,
    preprocess_from_dat,
    preprocess_kulfan_parameters,
    load_json,
    load_model_from_state_dict,
    load_scalers,
    find_file
)
from src.models_ import instantiate_model

from typing import Optional

set_seed(42)

MODEL_PATH, PARAMS_PATH = map(lambda x: find_file(file=x), ['cg_model.pt', 'cg_params.json'])
SCALER_DIR, DAT_DIR = map(lambda x: find_file(dir=x), ['cg_scalers', 'aerofoil_data'])
params = load_json(PARAMS_PATH)

class CurveGen:
    def __init__(self, k_params: Optional[dict] = None, dat_path: Optional[str] = None, coords: Optional[list] = None, aero_name: Optional[str] = None):
        """Initializes the model and preprocesses the input data."""
        # Load model
        self.model = instantiate_model(
            net_name=params['net_name'],
            input_size=params['input_size'],
            output_size=params['output_size'],
            params=params
        )
        load_model_from_state_dict(self.model, MODEL_PATH)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Load scalers
        self.X_scaler, self.y_scaler = load_scalers(SCALER_DIR)

        # Store input parameters
        self.k_params = k_params
        self.dat_path = dat_path
        self.coords = coords
        self.aero_name = aero_name
        self.y_pred = None

        # Process inputs and generate the curve
        self._preprocess()
        self._generate_curve()

    def _preprocess(self):
        """Handles the aerofoil input preprocessing."""
        if self.dat_path is not None:
            self.k_params, self.aero_name, self.coords = preprocess_from_dat(file_path=self.dat_path, dat_directory=DAT_DIR)
            print(f"Loaded {self.aero_name} from {self.dat_path}")
        elif self.aero_name is not None:
            self.k_params, self.aero_name, self.coords = preprocess_from_dat(aerofoil_name=self.aero_name, dat_directory=DAT_DIR)
            print(f"Loaded {self.aero_name}")
        elif self.coords is not None:
            self.k_params = preprocess_from_dat(coords=self.coords, dat_directory=DAT_DIR)
            print('Loaded coordinates')
        elif self.k_params is not None:
            pass
        else:
            raise ValueError("Either dat_path, aero_name, or coords must be provided.")

    def _generate_curve(self):
        """Processes the input, scales it, runs the model, and inverse transforms the predictions."""
        # Convert Kulfan parameters into structured input
        self.X_input = preprocess_kulfan_parameters(self.k_params)
        self.X_input = np.array(self.X_input).reshape(1, -1)

        # Apply the same feature scaling as during training (DO NOT FIT AGAIN!)
        X_scaled = self.X_scaler.transform(self.X_input)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # Ensure GRU model input shape is correct
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)

        # Generate model prediction
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor).cpu().numpy()

        # Convert predictions back to original scale
        self.y_pred = self.y_scaler.inverse_transform(y_pred_scaled).flatten()

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
            raise ValueError("No predictions available. Ensure `_generate_curve()` has been executed.")

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

