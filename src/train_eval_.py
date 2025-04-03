import torch
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error, mean_absolute_error
from src.losses_ import physics_informed_loss, mse_loss, huber_loss, logcosh_loss
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

def train_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader, num_epochs: int = 400, 
                scheduler: torch.optim.lr_scheduler._LRScheduler = None, 
                loss_func_name='mse', patience: int = 40, save: str = None, 
                l2: bool = None) -> Tuple[List[float], List[float]]:

    """
    Train a PyTorch model with early stopping.

    Args:
        model: Model to be trained.
        optimizer: Specified optimizer for training.
        train_loader: Trainloader.
        test_loader: Test loader.
        num_epochs: Number of runs for training loop.
        scheduler: (optional)
        loss_func_name: Loss function name, defaults to 'mse'. One of 'mse', 'huber', 'log_cosh', 'pi'.
        patience: Patience for early stopping.
        save (str): Full filepath of model to save, if None, don't save (optional)
        l2 (Bool): Whether to use L2 regularization (optional)

    Returns:
        Trained model, ready to be evaluated.
        tuple: (train_losses, val_losses)
    """
    loss_functions = {
        "huber": huber_loss,
        "mse": mse_loss,
        "log_cosh": logcosh_loss,
        "pi": physics_informed_loss,
    }
    loss_func = loss_functions.get(loss_func_name.lower(), mse_loss)
    device = next(model.parameters()).device
    early_stop_counter = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_func(y_pred, y_batch)
            if l2:
                l2_norm = sum(p.pow(2).sum() for p in model.parameters())
                loss += 0.0000001 * l2_norm
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss += loss_func(y_pred, y_batch).item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        if scheduler:
            scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            if save:
                torch.save(model.state_dict(), save)
        else:
            early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
    return train_losses, val_losses


def evaluate_model(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, y_scaler: any) -> dict:
    """
    Evaluate a PyTorch model.
    """
    device = next(model.parameters()).device
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            predictions.append(y_pred)
            targets.append(y_batch.cpu().numpy())
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    predictions = y_scaler.inverse_transform(predictions)
    targets = y_scaler.inverse_transform(targets)
    results = {
        'predictions': predictions,
        'targets': targets,
        'r2': r2_score(targets, predictions),
        'mse': mean_squared_error(targets, predictions),
        'rmse': root_mean_squared_error(targets, predictions),
        'mae': mean_absolute_error(targets, predictions)
    }
    
    print(f"\nR² Score: {results['r2']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print(f"Root Mean Squared Error: {results['rmse']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f}")
    return results
