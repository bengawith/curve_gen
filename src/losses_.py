import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LogCoshLoss(nn.Module):
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(y_pred - y_true))
        return loss.mean()

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        error = y_true - y_pred
        is_small_error = torch.abs(error) <= self.delta
        squared_loss = 0.5 * (error ** 2)
        linear_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        return torch.mean(torch.where(is_small_error, squared_loss, linear_loss))


def logcosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    criterion = LogCoshLoss()
    return criterion(y_pred, y_true)


def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    criterion = HuberLoss(delta=delta)
    return criterion(y_pred, y_true)


def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    criterion = nn.MSELoss()
    return criterion(y_pred, y_true)


def physics_informed_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
           expected_slope: float = 0.1,
           lambda_monotonicity: float = 0.05,
           lambda_smooth: float = 0.01,
           lambda_zero: float = 0.05) -> torch.Tensor:
    """
    Adjusted Physics-Informed Lift Loss Using Unscaled Predictions

    Assumptions:
      - y_pred has 96 columns: first 48 are C_L predictions and last 48 are angle predictions (α), all in a scaled space.
      - A combined scaler was used when fitting the training targets.
    
    This loss consists of:
      1. MSE loss between unscaled predicted and true C_L.
      2. Monotonicity loss enforcing dC_L/dα >= expected_slope.
      3. Smoothness loss penalizing large second derivatives.
      4. Zero-lift loss enforcing that C_L is near zero at the angle closest to zero.
      
    Note: If possible, consider using separate scalers for C_L and α.
    """

    # 1. Load the saved scaler.
    scaler = joblib.load(r"C:\Users\benga\workspaces\github.com\bengawith\project\final_model\scalers\y_scaler.gz")
    
    # Convert scaler parameters to torch tensors.
    scale = torch.tensor(scaler.scale_, dtype=y_pred.dtype, device=y_pred.device)
    min_val = torch.tensor(scaler.min_, dtype=y_pred.dtype, device=y_pred.device)

    # 2. Extract scaled predictions and targets.
    y_pred_CL_scaled = y_pred[:, :48]
    y_true_CL_scaled = y_true[:, :48]
    y_pred_alpha_scaled = y_pred[:, 48:]

    # 3. Unscale into physical units.
    y_pred_CL = y_pred_CL_scaled * scale[:48] + min_val[:48]
    y_true_CL = y_true_CL_scaled * scale[:48] + min_val[:48]
    y_pred_alpha = y_pred_alpha_scaled * scale[48:] + min_val[48:]

    # 4. Compute MSE loss between unscaled C_L values.
    mse_loss = torch.mean((y_true_CL - y_pred_CL) ** 2)

    # 5. Monotonicity Loss: enforce dC_L/dα >= expected_slope.
    delta_alpha = y_pred_alpha[:, 1:] - y_pred_alpha[:, :-1]
    dCL_dalpha = (y_pred_CL[:, 1:] - y_pred_CL[:, :-1]) / delta_alpha
    monotonicity_loss = torch.mean(torch.relu(expected_slope - dCL_dalpha) ** 2)
    
    # 6. Smoothness Loss: penalize large second derivatives.
    dCL_dalpha_diff = dCL_dalpha[:, 1:] - dCL_dalpha[:, :-1]
    delta_alpha_mid = (delta_alpha[:, :-1] + delta_alpha[:, 1:]) / 2
    d2CL_dalpha2 = dCL_dalpha_diff / delta_alpha_mid
    smoothness_loss = torch.mean(d2CL_dalpha2 ** 2)
    
    # 7. Zero-Lift Loss: enforce C_L near zero at the angle closest to zero.
    zero_idx = torch.argmin(torch.abs(y_pred_alpha[0]))
    zero_lift_loss = torch.mean(y_pred_CL[:, zero_idx] ** 2)
    
    # 8. Total Loss: combine the terms.
    total_loss = (mse_loss +
                  lambda_monotonicity * monotonicity_loss +
                  lambda_smooth * smoothness_loss +
                  lambda_zero * zero_lift_loss)
    
    return total_loss