import sys
import os
import numpy as np
import torch

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from curve_gen.curve_gen import CurveGen, CurveGenError
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from src.utils_ import read_coords

import matplotlib
import tempfile
import json
from unittest.mock import patch, MagicMock
matplotlib.use('Agg')

coords = read_coords('./data/aerofoil_data/naca2410.dat')
k_params = get_kulfan_parameters(coordinates=coords, n_weights_per_side=6)

class TestCurveGenValidation(unittest.TestCase):
    """Test input validation in CurveGen."""

    def test_no_inputs_raises_error(self):
        with self.assertRaises(CurveGenError) as cm:
            CurveGen()
        self.assertIn("Exactly one input", str(cm.exception))

    def test_multiple_inputs_raises_error(self):
        with self.assertRaises(CurveGenError) as cm:
            CurveGen(k_params={}, coords=[])
        self.assertIn("Exactly one input", str(cm.exception))

    def test_valid_single_input_k_params(self):
        # Assuming k_params is valid
        cg = CurveGen(k_params=k_params)
        self.assertIsNotNone(cg.k_params)

    def test_valid_single_input_coords(self):
        cg = CurveGen(coords=coords)
        self.assertIsNotNone(cg.coords)

    def test_valid_single_input_aero_name(self):
        cg = CurveGen(aero_name='naca2410')
        self.assertEqual(cg.aero_name, 'naca2410')

    def test_valid_single_input_dat_path(self):
        cg = CurveGen(dat_path='./data/aerofoil_data/naca2410.dat')
        self.assertIsNotNone(cg.dat_path)


class TestCurveGenPathsSetup(unittest.TestCase):
    """Test path setup and file existence checks."""

    @patch('curve_gen.curve_gen.Path.is_file')
    def test_missing_model_file_raises_error(self, mock_is_file):
        mock_is_file.return_value = False
        with self.assertRaises(CurveGenError) as cm:
            CurveGen(k_params=k_params)
        self.assertIn("Model file not found", str(cm.exception))

    @patch('curve_gen.curve_gen.Path.is_file')
    def test_missing_params_file_raises_error(self, mock_is_file):
        def side_effect():
            # This is called as a property/method on the Path instance
            # We need to check the path from the call stack or use different approach
            return False  # Always return False for params file
        
        # Set up the mock to return True for model file, False for params file
        call_count = 0
        def side_effect_with_count():
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call is for model file
                return True
            return False  # Second call is for params file
            
        mock_is_file.side_effect = side_effect_with_count
        with self.assertRaises(CurveGenError) as cm:
            CurveGen(k_params=k_params)
        self.assertIn("Parameters file not found", str(cm.exception))


class TestCurveGenModelInitialization(unittest.TestCase):
    """Test model and scaler initialization."""

    @patch('curve_gen.curve_gen.load_json')
    @patch('curve_gen.curve_gen.instantiate_model')
    @patch('curve_gen.curve_gen.torch.load')
    @patch('curve_gen.curve_gen.load_scalers')
    def test_model_initialization_success(self, mock_load_scalers, mock_torch_load, mock_instantiate, mock_load_json):
        mock_load_json.return_value = {'net_name': 'test', 'input_size': 10, 'output_size': 20}
        # Create a proper mock model that returns tensors
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[1.0, 2.0]])  # Return a tensor instead of MagicMock
        mock_instantiate.return_value = mock_model
        mock_torch_load.return_value = {}
        
        # Create mock scalers with proper transform methods
        mock_X_scaler = MagicMock()
        mock_X_scaler.transform.return_value = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])  # 13 features
        mock_y_scaler = MagicMock()
        mock_y_scaler.inverse_transform.return_value = np.array([[1.0, 2.0]])
        mock_load_scalers.return_value = (mock_X_scaler, mock_y_scaler)
        
        cg = CurveGen(k_params=k_params)
        self.assertIsNotNone(cg.model)

    @patch('curve_gen.curve_gen.load_json', side_effect=Exception("Load error"))
    def test_model_initialization_failure(self, mock_load_json):
        with self.assertRaises(CurveGenError) as cm:
            CurveGen(k_params=k_params)
        self.assertIn("Error initializing model or scalers", str(cm.exception))


class TestCurveGenCache(unittest.TestCase):
    """Test caching functionality."""

    def setUp(self):
        self.cg = CurveGen(k_params=k_params)

    def test_cache_setup(self):
        self.assertIsNotNone(self.cg._prediction_cache)

    def test_generate_uses_cache(self):
        # Call generate twice with same input to check caching
        result1 = self.cg._generate()
        result2 = self.cg._generate()
        
        # Results should be identical (indicating cache usage)
        np.testing.assert_array_equal(result1, result2)
        
        # Check that cache has been set up
        self.assertIsNotNone(self.cg._prediction_cache)


class TestCurveGenPreprocessing(unittest.TestCase):
    """Test input preprocessing."""

    def setUp(self):
        self.cg = CurveGen(k_params=k_params)

    @patch('curve_gen.curve_gen.preprocess_from_dat')
    def test_extract_from_dat_path(self, mock_preprocess):
        mock_preprocess.return_value = ({}, 'test', [])
        input_key = json.dumps({"dat_path": "test.dat"}, sort_keys=True)
        k_params, aero_name, coords = self.cg._extract_params_from_key(input_key)
        mock_preprocess.assert_called_with(file_path="test.dat", dat_directory=str(self.cg.dat_dir))

    @patch('curve_gen.curve_gen.get_kulfan_parameters')
    def test_extract_from_coords(self, mock_get_kulfan):
        mock_get_kulfan.return_value = {}
        input_key = json.dumps({"coords": [[0,0], [1,1]]}, sort_keys=True)
        k_params, aero_name, coords = self.cg._extract_params_from_key(input_key)
        mock_get_kulfan.assert_called_once()

    def test_preprocess_input_updates_attributes(self):
        # Create a proper k_params structure for testing
        test_k_params = {
            'lower_weights': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            'upper_weights': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 
            'TE_thickness': 0.01,
            'leading_edge_weight': 0.5
        }
        input_key = json.dumps({"k_params": {
            'lower_weights': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            'upper_weights': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 
            'TE_thickness': 0.01,
            'leading_edge_weight': 0.5
        }}, sort_keys=True)
        tensor = self.cg._preprocess_input(input_key)
        self.assertIsInstance(tensor, torch.Tensor)


class TestCurveGenGeneration(unittest.TestCase):
    """Test prediction generation."""

    def setUp(self):
        self.cg = CurveGen(k_params=k_params)

    def test_generate_returns_array(self):
        result = self.cg._generate()
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(len(result), 0)


class TestCurveGenDataRetrieval(unittest.TestCase):
    """Test get_data method."""

    def setUp(self):
        self.cg = CurveGen(k_params=k_params)

    def test_get_data_structure(self):
        data = self.cg.get_data()
        self.assertIn('aero_name', data)
        self.assertIn('k_params', data)
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], np.ndarray)


class TestCurveGenPlotting(unittest.TestCase):
    """Test plotting methods."""

    def setUp(self):
        self.cg_kulf = CurveGen(k_params=k_params)
        self.cg_coords = CurveGen(coords=coords)

    def test_plot_curve_without_predictions_raises_error(self):
        cg = CurveGen.__new__(CurveGen)  # Create without init
        cg.y_pred = None
        with self.assertRaises(CurveGenError):
            cg.plot_curve()

    def test_plot_aero_without_coords_raises_error(self):
        cg = CurveGen.__new__(CurveGen)
        cg.coords = None
        with self.assertRaises(CurveGenError):
            cg.plot_aero()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_curve_with_save_path(self, mock_show, mock_savefig):
        self.cg_kulf.plot_curve(save_path='test.png')
        mock_savefig.assert_called_with('test.png')

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_aero_with_save_path(self, mock_show, mock_savefig):
        self.cg_coords.plot_aero(save_path='test.png')
        mock_savefig.assert_called_with('test.png')


class TestCurveGenSaving(unittest.TestCase):
    """Test save_data method."""

    def setUp(self):
        self.cg = CurveGen(k_params=k_params)

    def test_save_data_creates_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'test.json')
            self.cg.save_data(file_path)
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.assertIn('predictions', data)


class TestCurveGenEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_coords(self):
        with self.assertRaises(Exception):  # Depending on get_kulfan_parameters
            CurveGen(coords=[])

    @patch('curve_gen.curve_gen.get_kulfan_parameters', side_effect=Exception("Invalid params"))
    def test_invalid_k_params(self, mock_get_kulfan):
        with self.assertRaises(Exception):
            CurveGen(coords=[[0,0], [1,1]])

    def test_cache_size_zero(self):
        cg = CurveGen(k_params=k_params, cache_size=0)
        self.assertEqual(cg._cache_size, 0)

    @patch('curve_gen.curve_gen.torch.cuda.is_available', return_value=False)
    def test_device_cpu(self, mock_cuda):
        cg = CurveGen(k_params=k_params)
        self.assertEqual(cg.device, torch.device('cpu'))

if __name__ == '__main__':
    unittest.main()