import sys
sys.path.append('./')

import json
from time import time
from src.data_ import DataPrep
from src.train_eval_ import train_model, evaluate_model
from src.models_ import instantiate_model, adam
from src.utils_ import set_seed, run_time

set_seed(42)

gru_params = json.load(open('./optuna_results/log_cosh/GRUModel.json'))
lstm_params = json.load(open('./optuna_results/pi/LSTMModel.json'))

dp = DataPrep('./data/csv/14KP_48CLA.csv')
data = dp.get_data()

gru_model = instantiate_model(
    net_name='GRUModel',
    input_size=data['input_size'],
    output_size=data['output_size'],
    params=gru_params['0']
)
optimiser_gru = adam(gru_model, gru_params['0']['learning_rate'])

lstm_model = instantiate_model(
    net_name='LSTMModel',
    input_size=data['input_size'],
    output_size=data['output_size'],
    params=lstm_params['0']
)
optimiser_lstm = adam(lstm_model, lstm_params['0']['learning_rate'])

train_dict = {
    "GRU": {
        "model": gru_model,
        "optim": optimiser_gru,
        "lf": 'log_cosh',
        "params": gru_params['0']
    },
    "LSTM": {
        "model": lstm_model,
        "optim": optimiser_lstm,
        "lf": 'pi',
        "params": lstm_params['0']
    }
}

results = {}
for model in train_dict:
    start = time()
    _, _ = train_model(
        model=train_dict[model]['model'],
        optimizer=train_dict[model]['optim'],
        train_loader=data['train_loader'],
        val_loader=data['test_loader'],
        num_epochs=train_dict[model]['params']['num_epochs']
    )

    res = evaluate_model(
        train_dict[model]['model'],
        test_loader=data['test_loader'],
        y_scaler=data['y_scaler']
    )

    end = time()

    rt = run_time(start, end, ret=True)

    res['run_time'] = rt
    results[model] = res

print(results)

json.dump(results, open('time_test.json', 'w'), indent=4)

results = {
    'GRU': {
        'r2': 0.771634578704834, 
        'mse': 0.8435173630714417, 
        'rmse': 0.7101494669914246, 
        'mae': 0.503777801990509, 
        'run_time': 107.08138751983643
    }, 
    'LSTM': {
        'r2': 0.7733926773071289, 
        'mse': 0.8109087347984314, 
        'rmse': 0.699310839176178, 
        'mae': 0.49568215012550354, 
        'run_time': 216.26096057891846
    }
}