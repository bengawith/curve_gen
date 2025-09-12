import time
import numpy as np
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from curve_gen.curve_gen import CurveGen

# Dummy test data for fast tests (replace with real .dat or params for full test)
DUMMY_K_PARAMS = {str(i): 0.1 for i in range(12)}
DUMMY_COORDS = [[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]]
DUMMY_AERO_NAME = "a18"
DUMMY_DAT_PATH = None  # Set to a valid .dat path for integration test


def test_curvegen_prediction_and_caching(monkeypatch):
    """Test prediction, caching, and timing."""
    # Patch model/scaler loading to avoid heavy computation if needed
    # (Assume real files exist for integration test)
    cg = CurveGen(k_params=DUMMY_K_PARAMS, coords=DUMMY_COORDS, aero_name=DUMMY_AERO_NAME, cache_size=2)
    # First prediction (should not be cached)
    start = time.perf_counter()
    y1 = cg._generate()
    t1 = time.perf_counter() - start
    assert isinstance(y1, np.ndarray)
    # Second prediction with same input (should be cached, much faster)
    start = time.perf_counter()
    y2 = cg._generate()
    t2 = time.perf_counter() - start
    assert np.allclose(y1, y2)
    assert t2 < t1 * 0.5 or t2 < 0.01  # Cached call should be much faster
    # Third prediction with new input (not cached)
    cg2 = CurveGen(k_params={str(i): 0.2 for i in range(12)}, coords=DUMMY_COORDS, aero_name="ag03", cache_size=2)
    y3 = cg2._generate()
    assert not np.allclose(y1, y3)
    # Fill cache and check eviction
    cg._generate()  # 1st input
    cg2._generate() # 2nd input
    cg3 = CurveGen(k_params={str(i): 0.3 for i in range(12)}, coords=DUMMY_COORDS, aero_name="ag04", cache_size=2)
    cg3._generate() # 3rd input, should evict oldest
    assert len(cg3._prediction_cache) <= 2

def test_curvegen_get_data():
    cg = CurveGen(k_params=DUMMY_K_PARAMS, coords=DUMMY_COORDS, aero_name=DUMMY_AERO_NAME)
    data = cg.get_data()
    assert 'aero_name' in data and 'k_params' in data and 'predictions' in data
    assert isinstance(data['predictions'], np.ndarray)

def test_curvegen_plot_methods(monkeypatch):
    cg = CurveGen(k_params=DUMMY_K_PARAMS, coords=DUMMY_COORDS, aero_name=DUMMY_AERO_NAME)
    # Patch plt.show to avoid blocking
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    cg.plot_curve()
    cg.plot_aero()

def test_curvegen_invalid_paths():
    import os
    from pathlib import Path
    # Use a non-existent model_dir
    bad_dir = str(Path(__file__).parent / "nonexistent_dir")
    with pytest.raises(FileNotFoundError):
        CurveGen(k_params=DUMMY_K_PARAMS, coords=DUMMY_COORDS, aero_name=DUMMY_AERO_NAME, model_dir=bad_dir)

def test_curvegen_no_coords():
    cg = CurveGen(k_params=DUMMY_K_PARAMS, coords=None, aero_name=DUMMY_AERO_NAME)
    cg.coords = None
    with pytest.raises(ValueError):
        cg.plot_aero()

def test_curvegen_no_predictions():
    cg = CurveGen(k_params=DUMMY_K_PARAMS, coords=DUMMY_COORDS, aero_name=DUMMY_AERO_NAME)
    cg.y_pred = None
    with pytest.raises(ValueError):
        cg.plot_curve()
