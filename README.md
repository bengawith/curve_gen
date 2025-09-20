# CurveGen: Neural Network-Based Airfoil Lift Curve Prediction

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.12+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A sophisticated neural network-based tool for predicting airfoil lift coefficient (CL) vs. angle of attack curves from airfoil geometries. CurveGen uses GRU-based neural networks trained on extensive aerodynamic data to provide accurate lift curve predictions across pre-stall, stall, and post-stall regions.

## 🌟 Key Features

- **Multiple Input Methods**: Supports Kulfan parameters, .dat files, coordinate arrays, and airfoil names
- **Advanced Neural Architecture**: GRU-based model with optimized hyperparameters
- **Comprehensive Output**: Predicts 48 CL values and 48 angle-of-attack values
- **Professional CLI**: Full-featured command-line interface with extensive options
- **Robust Testing**: Comprehensive test suite with 26+ test cases
- **Caching System**: LRU caching for improved performance
- **Data Export**: JSON export with numpy array support
- **Visualization**: Customizable plotting for lift curves and airfoil shapes

## 🚀 Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/bengawith/curve_gen.git
   cd curve_gen
   ```

2. **Set up the environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

#### Using the Python API
```python
from curve_gen.curve_gen import CurveGen

# Using an airfoil name
cg = CurveGen(aero_name='naca2410')
cg.plot_curve()
cg.plot_aero()

# Save analysis data
cg.save_data('naca2410_analysis.json')
```

#### Using the Command Line Interface
```bash
# Basic prediction
python cg.py --aero_name "naca2410"

# Advanced usage with custom output
python cg.py --dat_path "./data/aerofoil_data/clarkx.dat" \
             --save_plots --save_data --verbose \
             --output_dir "./results"

# Batch processing
python cg.py --aero_name "ag19" --data_only --no_plots
```

#### Running Examples
```bash
# Comprehensive examples demonstrating all features
python main.py
```

## 📁 Project Structure

```
curve_gen/
├── 📁 curve_gen/              # Core CurveGen module
│   ├── curve_gen.py          # Main CurveGen class
│   ├── cg_model.pt           # Trained neural network model state dict
│   ├── cg_params.json        # Model parameters and configuration
│   └── cg_scalers/           # Input/output data scalers
│       ├── X_scaler.gz       # Input feature scaler
│       └── y_scaler.gz       # Output target scaler
├── 📁 data/                  # Training and reference data
│   ├── aerofoil_data/        # Airfoil geometry files (.dat format)
│   └── csv/                  # Processed datasets
├── 📁 src/                   # Source utilities and modules
│   ├── data_.py              # Data preprocessing utilities
│   ├── models_.py            # Neural network architectures
│   ├── utils_.py             # General utility functions
│   ├── train_eval_.py        # Training and evaluation logic
│   ├── losses_.py            # Custom loss functions
│   ├── errors_.py            # Error metrics
│   ├── plot_.py              # Plotting utilities
│   └── tuning_.py            # Hyperparameter optimization
├── 📁 tests/                 # Comprehensive test suite
│   ├── test_curve_gen.py     # CurveGen class tests (26+ tests)
│   ├── test_codebase.py      # General codebase tests
│   └── test_curve_gen_robust.py # Robustness tests
├── 📁 exploration/           # Research and development notebooks
│   ├── data.ipynb            # Data analysis
│   ├── final_analysis.ipynb  # Final model analysis
│   ├── gru_train.ipynb       # Model training experiments
│   └── optim_kp.ipynb        # Hyperparameter optimization
├── 📁 scripts/               # Utility scripts
│   ├── full_set_gen.py       # Dataset generation
│   ├── main_tuning.py        # Model tuning
│   └── sensitivity_analysis.py # Model sensitivity analysis
├── 📁 optuna_results/        # Hyperparameter optimization results
├── 📁 kp_results/            # Kulfan parameter analysis results
├── 📁 paper/                 # Research documentation
│   └── final_report.pdf      # Complete project report
├── main.py                   # Comprehensive usage examples
├── cg.py                     # Professional CLI interface
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 CurveGen API Reference

### Class Initialization
```python
CurveGen(
    k_params=None,      # Kulfan parameters (dict)
    dat_path=None,      # Path to .dat file (str)
    coords=None,        # Coordinate array (list)
    aero_name=None,     # Airfoil name (str)
    cache_size=128,     # LRU cache size (int)
    model_dir=None      # Custom model directory (str)
)
```

### Key Methods
- **`plot_curve(save_path=None, **kwargs)`**: Generate lift curve visualization
- **`plot_aero(save_path=None, **kwargs)`**: Generate airfoil shape visualization
- **`get_data()`**: Retrieve prediction data and metadata
- **`save_data(file_path)`**: Export analysis to JSON file

### Input Options (choose one)

#### 1. Kulfan Parameters
```python
k_params = {
    'lower_weights': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'upper_weights': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'TE_thickness': 0.01,
    'leading_edge_weight': 0.5
}
cg = CurveGen(k_params=k_params)
```

#### 2. DAT File Path
```python
cg = CurveGen(dat_path='./data/aerofoil_data/naca2410.dat')
```

#### 3. Coordinate Arrays
```python
coords = [[x1, y1], [x2, y2], [x3, y3], ...]
cg = CurveGen(coords=coords)
```

#### 4. Airfoil Name
```python
cg = CurveGen(aero_name='naca2410')  # From database
```

## 💻 Command Line Interface

The `cg.py` script provides a comprehensive CLI with extensive options:

### Basic Commands
```bash
# Simple prediction
python cg.py --aero_name "naca2410"

# Using DAT file
python cg.py --dat_path "./data/aerofoil_data/clarkx.dat"

# Using coordinates
python cg.py --coords "[[0,0],[0.5,0.1],[1,0]]"

# Using Kulfan parameters
python cg.py --k_params '{"lower_weights":[0.1,0.2,0.3,0.4,0.5,0.6],"upper_weights":[0.1,0.2,0.3,0.4,0.5,0.6],"TE_thickness":0.01,"leading_edge_weight":0.5}'
```

### Advanced Options
```bash
# Save outputs with custom styling
python cg.py --aero_name "ag19" \
             --save_plots --save_data \
             --output_dir "./results" \
             --curve_color "red" \
             --aero_color "blue" \
             --verbose

# Batch processing
python cg.py --dat_path "airfoil.dat" \
             --data_only --no_plots \
             --cache_size 256

# Custom model directory
python cg.py --aero_name "naca0012" \
             --model_dir "./custom_models" \
             --plot_style "seaborn"
```

### CLI Options Summary
- **Input types**: `--k_params`, `--dat_path`, `--coords`, `--aero_name`
- **Output control**: `--save_plots`, `--save_data`, `--output_dir`
- **Display options**: `--plot_only`, `--data_only`, `--no_plots`, `--verbose`
- **Customization**: `--curve_color`, `--aero_color`, `--plot_style`
- **Performance**: `--cache_size`, `--model_dir`

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_curve_gen.py -v          # CurveGen class tests
pytest tests/test_codebase.py -v           # General codebase tests
pytest tests/test_curve_gen_robust.py -v   # Robustness tests

# Run with coverage
pytest tests/ --cov=curve_gen --cov-report=html
```

**Test Coverage**: 26+ test cases covering:
- Input validation and error handling
- Model initialization and loading
- Caching functionality
- Data preprocessing and prediction
- Plotting and visualization
- Data export and serialization
- Edge cases and error conditions

## 📊 Model Architecture & Performance

### Neural Network Details
- **Architecture**: GRU-based recurrent neural network
- **Input**: 13 Kulfan parameters (6 lower weights + 6 upper weights + TE thickness)
- **Output**: 96 values (48 CL + 48 α predictions)
- **Training Data**: UIUC Airfoil Database + CFD simulations
- **Validation**: XFoil cross-validation

### Key Performance Metrics
- **Angle Range**: -15° to +18° (comprehensive stall modeling)
- **CL Range**: Accurate across pre-stall, stall, and post-stall regions
- **Prediction Speed**: <100ms per airfoil (with caching)
- **Memory Usage**: Optimized with LRU caching system

## 🔬 Research Background

This project implements research in neural network-based aerodynamic prediction:

- **Objective**: Predict complete lift curves from airfoil geometry
- **Innovation**: End-to-end learning from geometry to aerodynamic performance
- **Applications**: Rapid airfoil analysis, design optimization, educational tools
- **Validation**: Comprehensive comparison with XFoil and CFD results

For detailed methodology and results, see [`paper/final_report.pdf`](paper/final_report.pdf).

## 🛠️ Development

### Setting up Development Environment
```bash
# Clone and setup
git clone https://github.com/bengawith/curve_gen.git
cd curve_gen
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run examples
python main.py
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create Pull Request

## 📚 Examples and Tutorials

The [`main.py`](main.py) file contains comprehensive examples:

1. **Basic Usage**: Simple airfoil prediction
2. **Advanced Features**: Custom styling and data export
3. **Performance Demo**: Caching system demonstration
4. **Error Handling**: Robust error management
5. **Batch Processing**: Multiple airfoil analysis

## 🔗 Data Sources

- **[UIUC Airfoil Database](https://m-selig.ae.illinois.edu/ads/coord_database.html)**: Primary airfoil geometry source
- **[Airfoil Tools](http://airfoiltools.com/)**: Additional validation data
- **CFD Simulations**: Ansys Fluent validation dataset

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ben Gawith** - [enrbgawi@ljmu.ac.uk](mailto:ben@gawith.com)

## 🙏 Acknowledgments

- UIUC Airfoil Database for comprehensive airfoil data
- AeroSandbox library for Kulfan parameter processing
- PyTorch team for the neural network framework
- Optuna for hyperparameter optimization tools

## 📈 Version History

- **v1.0.0**: Initial release with basic prediction capability
- **v2.0.0**: Enhanced CurveGen class with caching and error handling
- **v2.1.0**: Professional CLI interface and comprehensive testing
- **v2.2.0**: JSON export, plotting enhancements, and documentation

---

*For technical details, methodology, and validation results, please refer to the complete research paper in [`paper/final_report.pdf`](paper/final_report.pdf).*
*This work is being extended on in a paper to be published in the Elsevier Aerospace Science and Technology journal*