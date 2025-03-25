import optuna
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.models_ import FullyConnectedNN, GRUModel, LSTMModel
from src.losses_ import huber_loss, logcosh_loss, mse_loss, physics_informed_loss
from src.train_eval_ import train_model
from src.db import DataDB
import numpy as np
import atexit
import signal
import json
from typing import Any



def objective(trial: optuna.Trial, model_class: Any, input_size: int, output_size: int, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, loss_function_name: str = "huber", metric: str = "r2", device: str = "cuda") -> float:
    loss_functions = {
        "huber": huber_loss,
        "mse": mse_loss,
        "log_cosh": logcosh_loss,
        "pi": physics_informed_loss,
    }
    loss_func = loss_functions.get(loss_function_name.lower(), mse_loss)
    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 512, step=32) for i in range(num_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 300, 600)
    model = model_class(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, loss_func_name=loss_function_name)
    model.eval()
    val_targets, val_predictions = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_predictions.append(y_pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    if metric.lower().strip() == "mse":
        return mean_squared_error(val_targets, val_predictions)
    if metric.lower().strip() == "mae":
        return mean_absolute_error(val_targets, val_predictions)
    return r2_score(val_targets, val_predictions)

def tune_model(model_class: Any, input_size: int, output_size: int, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, 
               loss_function_name: str = "huber", num_trials: int = 100, metric: str = "r2", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> optuna.study.Study:
    metric = metric.strip().lower()
    study = optuna.create_study(direction="minimize" if metric in ["mse", "mae"] else "maximize")
    study.optimize(lambda trial: objective(trial, model_class, input_size, output_size, train_loader, val_loader, loss_function_name, metric, device), n_trials=num_trials)
    print("Best trial:")
    print(f"  {metric}: {study.best_trial.value}")
    print("  Hyperparameters: ", study.best_trial.params)
    return study


# Updated Objective Functions
def objective_fc(trial, input_size, output_size, train_loader, val_loader, device, loss_fn):
    num_layers = trial.suggest_int("num_layers", 2, 8)
    hidden_sizes = [trial.suggest_int(f"hidden_size_{i}", 32, 512, step=32) for i in range(num_layers)]
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 400, 600)
    
    model = FullyConnectedNN(input_size=input_size, hidden_sizes=hidden_sizes,
                             output_size=output_size, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                num_epochs=num_epochs, loss_func_name=loss_fn)
    model.eval()
    val_targets, val_predictions = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            val_predictions.append(y_pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    mse_val = np.mean((val_targets - val_predictions) ** 2)
    return mse_val


def objective_lstm(trial, input_size, output_size, train_loader, val_loader, device, loss_fn):
    hidden_size = trial.suggest_int("hidden_size", 32, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 8)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 400, 600)
    
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                      num_layers=num_layers, dropout=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Let the model handle unsqueezing if needed.
    train_model(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                num_epochs=num_epochs, loss_func_name=loss_fn)
    model.eval()
    val_targets, val_predictions = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)
            val_predictions.append(y_pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    mse_val = np.mean((val_targets - val_predictions) ** 2)
    return mse_val


def objective_gru(trial, input_size, output_size, train_loader, val_loader, device, loss_fn):
    hidden_size = trial.suggest_int("hidden_size", 32, 512, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 8)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 400, 600)
    
    model = GRUModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size,
                     num_layers=num_layers, dropout=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model=model, optimizer=optimizer, train_loader=train_loader, val_loader=val_loader,
                num_epochs=num_epochs, loss_func_name=loss_fn)
    model.eval()
    val_targets, val_predictions = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            if X_batch.dim() == 2:
                X_batch = X_batch.unsqueeze(1)
            y_pred = model(X_batch)
            val_predictions.append(y_pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())
    val_predictions = np.vstack(val_predictions)
    val_targets = np.vstack(val_targets)
    mse_val = np.mean((val_targets - val_predictions) ** 2)
    return mse_val


class TrialSaver:
    def __init__(self, study: optuna.study.Study, metric: str, file_path: str, top_n: int, maximize: bool = True) -> None:
        self.study = study
        self.metric = metric
        self.file_path = file_path
        self.top_n = top_n
        self.maximize = maximize
        atexit.register(self.save_on_exit)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)


    def save_on_exit(self, coll=None) -> None:
        print("\nSaving top trials before exit...")
        self.save_top_trials()


    def signal_handler(self, signum, frame=None) -> None:
        print(f"\nSignal {signum} received. Saving top trials before exit.")
        self.save_on_exit()
        exit(1)


    def save_top_trials_text(self) -> None:
        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else (float("-inf") if self.maximize else float("inf")),
            reverse=self.maximize
        )[:self.top_n]
        with open(self.file_path, "w") as file:
            file.write(f"Top {self.top_n} Trials ({self.metric}):\n\n")
            for i, trial in enumerate(sorted_trials, start=1):
                file.write(f"Rank {i}\n")
                file.write(f"  Trial Number: {trial.number}\n")
                file.write(f"  {self.metric.upper()}: {trial.value:.4f}\n")
                file.write("  Hyperparameters:\n")
                for param, value in trial.params.items():
                    file.write(f"    {param} = {value}\n")
                file.write("\n")
        print(f"Top {self.top_n} trials saved to {self.file_path}")


    def save_top_trials(self) -> None:
        if not isinstance(self.top_n, int) or self.top_n <= 0:
            raise ValueError(f"Invalid value for top_n: {self.top_n}. It must be a positive integer.")
        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else (float("-inf") if self.maximize else float("inf")),
            reverse=self.maximize
        )[:self.top_n]
        results = {}
        for i, trial in enumerate(sorted_trials):
            results[i] = trial.params
            results[i][self.metric] = trial.value
        with open(self.file_path, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Top {self.top_n} trials (JSON) saved to {self.file_path}")

    
    def save_to_db(self, collection: str = 'top_models') -> None:
        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else (float("-inf") if self.maximize else float("inf")),
            reverse=self.maximize
        )[:self.top_n]

        results = {}
        for i, trial in enumerate(sorted_trials):
            results[i] = trial.params
            results[i][self.metric] = trial.value

        DataDB().add_many(collection, [results[f'{i}'] for i in range(len(results))])


