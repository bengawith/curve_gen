import os
import torch
import optuna
import logging

from src.data_ import DataPrep
from src.models_ import instantiate_model
from src.tuning_ import TrialSaver, objective_fc, objective_gru, objective_lstm
from src.utils_ import set_seed, serialize_state_dict
from src.db import DataDB


# Set random seed for reproducibility
set_seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Map network names to their corresponding objective functions.
objective_functions = {
    "FullyConnectedNN": objective_fc,
    "LSTMModel": objective_lstm,
    "GRUModel": objective_gru,
}

# Map data preparation method names to their classes.
data_prep_classes = {
    "Standard": DataPrep,
    #"FeatureBinning": DataPrepFeatureBinning,
    #"DynamicBinning": DataPrepDynamicBinning,
}

# Define loss functions to iterate over.
loss_fns = ["log_cosh", "mse", "pi", "huber"]

# Main tuning loop

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Path to CSV dataset
    csv_file = "data/csv/14KP_48CLA.csv"
    
    # Number of trials for each study
    num_trials = 500
    
    # Base directory to save local trial results
    base_results_dir = "optuna_results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Initialize MongoDB interface
    db = DataDB()
    
    tuning_results = {}
    
    # Outer loop: iterate over loss functions
    for loss_name in loss_fns:
        logger.info(f"Starting tuning with loss function: {loss_name}")
        tuning_results[loss_name] = {}
        # Create subdirectory for this loss type
        loss_results_dir = os.path.join(base_results_dir, loss_name)
        os.makedirs(loss_results_dir, exist_ok=True)
        
        # Next loop: iterate over each data preparation method
        for dp_name, dp_class in data_prep_classes.items():
            logger.info(f"  DataPrep method: {dp_name}")
            dp_instance = dp_class(csv_file)
            data = dp_instance.get_data()
            train_loader = data["train_loader"]
            val_loader = data["test_loader"]
            input_size = data["input_size"]
            output_size = data["output_size"]
            
            if dp_name not in tuning_results[loss_name]:
                tuning_results[loss_name][dp_name] = {}
            
            # Next loop: iterate over each network model
            for net_name, objective_fn in objective_functions.items():
                logger.info(f"    Tuning network: {net_name} with loss {loss_name} and data {dp_name}")
                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: objective_fn(trial, input_size, output_size, train_loader, val_loader, device, loss_name),
                               n_trials=num_trials)
                best_trial = study.best_trial
                tuning_results[loss_name][dp_name][net_name] = best_trial.params
                logger.info(f"      Best hyperparameters: {best_trial.params}")
                
                # Save top trials locally in the subdirectory for this loss function
                results_filepath = os.path.join(loss_results_dir, f"{dp_name}_{net_name}_SS.json")
                trial_saver = TrialSaver(study=study, metric="mse", file_path=results_filepath, top_n=5, maximize=False)
                trial_saver.save_top_trials()

                # Re-instantiate the best model and obtain its state_dict
                best_model = instantiate_model(net_name, input_size, output_size, best_trial.params, device)
                state_dict = best_model.state_dict()
                serializable_state = serialize_state_dict(state_dict)

                # Build document to store in MongoDB
                document = {
                    "data_prep": dp_name,
                    "network": net_name,
                    "loss_fn": loss_name,
                    "best_params": best_trial.params,
                    "best_mse": best_trial.value,
                    "k_params": 14,
                    "model_state_dict": serializable_state
                }
                db.add_one(DataDB.TUNE_COLL, document)
                logger.info(f"  Document stored in MongoDB for {dp_name} - {net_name} with loss {loss_name}")
                
    
    print("\nTuning Results Summary:")
    for loss_name, dp_results in tuning_results.items():
        print(f"\nLoss Function: {loss_name}")
        for dp_name, net_dict in dp_results.items():
            print(f"  DataPrep Method: {dp_name}")
            for net_name, params in net_dict.items():
                print(f"    {net_name}: {params}")

if __name__ == "__main__":
    main()
