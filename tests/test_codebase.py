import unittest
import torch
import numpy as np
import tempfile
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from src.models_ import FullyConnectedNN, CNNModel, GRUModel, LSTMModel
from src.errors_ import mean_squared_error, r2_score
from src.losses_ import LogCoshLoss, HuberLoss, mse_loss
from src.data_ import DataPrep
from src.train_eval_ import evaluate_model
from src.plot_ import CurveVisualiser

class TestTorchModels(unittest.TestCase):
    def test_fully_connected_nn_forward(self):
        model = FullyConnectedNN(input_size=10, hidden_sizes=[20, 30], output_size=5)
        x = torch.randn(8, 10)
        out = model(x)
        self.assertEqual(out.shape, (8, 5))
    
    def test_gru_model_forward(self):
        model = GRUModel(input_size=14, hidden_size=64, output_size=96, num_layers=2)
        x = torch.randn(8, 14)
        out = model(x)
        self.assertEqual(out.shape, (8, 96))
    
    def test_lstm_model_forward(self):
        model = LSTMModel(input_size=14, hidden_size=64, output_size=96, num_layers=2)
        x = torch.randn(8, 5, 14)
        out = model(x)
        self.assertEqual(out.shape, (8, 96))

class TestTorchErrors(unittest.TestCase):
    def test_mean_squared_error(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 4])
        mse_val = mean_squared_error(y_true, y_pred)
        self.assertAlmostEqual(mse_val, 1/3, places=5)
    
    def test_r2_score(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        r2_val = r2_score(y_true, y_pred)
        self.assertEqual(r2_val, 1.0)

class TestTorchLosses(unittest.TestCase):
    def test_log_cosh_loss(self):
        loss_func = LogCoshLoss()
        y_pred = torch.tensor([0.0, 1.0])
        y_true = torch.tensor([0.0, 2.0])
        loss = loss_func(y_pred, y_true)
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_huber_loss(self):
        loss_func = HuberLoss(delta=1.0)
        y_pred = torch.tensor([0.0, 1.0])
        y_true = torch.tensor([0.0, 2.0])
        loss = loss_func(y_pred, y_true)
        self.assertGreaterEqual(loss.item(), 0.0)

class TestDataPrep(unittest.TestCase):
    def test_data_prep(self):
        dummy_data = {}

        for i in range(6):
            dummy_data[f"lower_weight_{i}"] = [0.1, 0.2]
            dummy_data[f"upper_weight_{i}"] = [0.5, 0.6]
        dummy_data["TE_thickness"] = [0.9, 1.0]
        dummy_data["leading_edge_weight"] = [1.1, 1.2]

        for i in range(48):
            dummy_data[f"CL_{i}"] = [0.2, 0.3]
            dummy_data[f"alpha_{i}"] = [5, 10]
        df = pd.DataFrame(dummy_data)

        with tempfile.TemporaryDirectory() as tmpdirname:
            csv_path = os.path.join(tmpdirname, "14KP_48CLA.csv")
            df.to_csv(csv_path, index=False)
            dp = DataPrep(csv_path)
            data = dp.get_data()
            self.assertIn("X_train", data)
            self.assertIn("X_scaler", data)

class TestCurveVisualiser(unittest.TestCase):
    def test_curve_visualiser(self):
        predictions = np.linspace(0, 1, 96)
        targets = np.linspace(0, 1, 96) + 0.1
        vis = CurveVisualiser(predictions, targets)
        try:
            vis.single_plot_CurveGen("TestAerofoil")
        except Exception as e:
            self.fail(f"CurveVisualiser.single_plot_CurveGen raised an exception: {e}")

class TestEvaluateModel(unittest.TestCase):
    def test_evaluate_model(self):
        from torch.utils.data import DataLoader, TensorDataset
        model = FullyConnectedNN(10, [20], 5)
        X = torch.randn(16, 10)
        y = torch.randn(16, 5)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=4)
        class DummyScaler:
            def inverse_transform(self, arr):
                return arr
        dummy_scaler = DummyScaler()
        results = evaluate_model(model, loader, dummy_scaler)
        self.assertIn("r2", results)
        self.assertIn("mse", results)

if __name__ == "__main__":
    unittest.main()
