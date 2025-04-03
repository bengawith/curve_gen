import sys
import time
import os
import json
import pandas as pd
from src.utils_ import load_json, set_seed
from src.models_ import instantiate_model, adam
from src.train_eval_ import train_model, evaluate_model
from src.data_ import DataPrep

set_seed(42)

def process_file(data, loss_function, file_name, model_name_map, results):
    with open(os.path.join('14KP', loss_function, file_name)) as json_file:
        hyperparameters = json.load(json_file)
        model = instantiate_model(
            net_name=model_name_map[file_name],
            input_size=data['input_size'],
            output_size=data['output_size'],
            params=hyperparameters['0']
        )
        _, _ = train_model(
            model=model,
            optimizer=adam(model=model, learning_rate=hyperparameters['0']['learning_rate']),
            train_loader=data['train_loader'],
            val_loader=data['test_loader'],
            num_epochs=hyperparameters['0']['num_epochs'],
            loss_func_name=loss_function
        )
        metrics = evaluate_model(
            model=model,
            test_loader=data['test_loader'],
            y_scaler=data['y_scaler']
        )
        results[loss_function][model_name_map[file_name]] = metrics

def process_file_2(data, results):
    model_name_map = {
        'Standard_GRUModel_top_trials.json': 'GRUModel',
        'Standard_LSTMModel_top_trials.json': 'LSTMModel',
        'Standard_FullyConnectedNN_top_trials.json': 'FullyConnectedNN',
        'Standard_CNNModel_top_trials.json': 'CNNModel',
    }
    for loss_function in os.listdir('14KP'):
        if os.path.isdir(os.path.join('14KP', loss_function)):
            results[loss_function] = {}
            for file_name in os.listdir(os.path.join('14KP', loss_function)):
                if file_name.startswith('Standard'):
                    process_file(data, loss_function, file_name, model_name_map, results)

def clean_res(results, save=False):
    for loss_function in results:
        results[loss_function] = {k: v for k, v in results[loss_function].items() if not k.startswith('DynamicBinning') and not 'CNN' in k}
        for model_name in results[loss_function]:
            results[loss_function][model_name] = {k: v for k, v in results[loss_function][model_name].items() if k not in ['predictions', 'targets']}
            for metric in results[loss_function][model_name]:
                results[loss_function][model_name][metric] = round(results[loss_function][model_name][metric], 4)

    top_dict = {
        'pi': {
            'r2':{'score': 0, 'net': None},
            'mse': {'score': float('inf'), 'net': None},
            'rmse': {'score': float('inf'), 'net': None},
            'mae': {'score': float('inf'), 'net': None,}
        },
        'huber': {
            'r2':{'score': 0, 'net': None},
            'mse': {'score': float('inf'), 'net': None},
            'rmse': {'score': float('inf'), 'net': None},
            'mae': {'score': float('inf'), 'net': None,}
        },
        'mse': {
            'r2':{'score': 0, 'net': None},
            'mse': {'score': float('inf'), 'net': None},
            'rmse': {'score': float('inf'), 'net': None},
            'mae': {'score': float('inf'), 'net': None,}
        },
        'log_cosh': {
            'r2':{'score': 0, 'net': None},
            'mse': {'score': float('inf'), 'net': None},
            'rmse': {'score': float('inf'), 'net': None},
            'mae': {'score': float('inf'), 'net': None,}
        }
    }

    for loss_function in results:
        for model_name in results[loss_function]:
            if results[loss_function][model_name]['r2'] > top_dict[loss_function]['r2']['score']:
                top_dict[loss_function]['r2']['score'] = results[loss_function][model_name]['r2']
                top_dict[loss_function]['r2']['net'] = model_name
            if results[loss_function][model_name]['mse'] < top_dict[loss_function]['mse']['score']:
                top_dict[loss_function]['mse']['score'] = results[loss_function][model_name]['mse']
                top_dict[loss_function]['mse']['net'] = model_name
            if results[loss_function][model_name]['rmse'] < top_dict[loss_function]['rmse']['score']:
                top_dict[loss_function]['rmse']['score'] = results[loss_function][model_name]['rmse']
                top_dict[loss_function]['rmse']['net'] = model_name
            if results[loss_function][model_name]['mae'] < top_dict[loss_function]['mae']['score']:
                top_dict[loss_function]['mae']['score'] = results[loss_function][model_name]['mae'] 
                top_dict[loss_function]['mae']['net'] = model_name

    results['top_dict'] = top_dict

    if save:
        with open('model_tune.json', 'w') as f: 
            json.dump(results, f, indent=4)
    
    return results

if __name__ == '__main__':
    dp = DataPrep('../data/csv/14KP_48CLA.csv')
    data = dp.get_data()
    results = {}
    process_file_2(data, results)
    results = clean_res(results, save=True)
