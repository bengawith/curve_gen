# Airfoil Lift Curve Prediction using Neural Networks (CurveGen)

This project explores the use of neural networks to predict lift coefficient (CL) vs. angle of attack curves from aerofoil geometries using CST (Class Shape Transformation) parameters. It integrates both XFoil-generated data and CFD simulations (Ansys Fluent) to validate the accuracy and generalisation of the model. Please refer to "paper/Final_Report.pdf" for the full project paper.

## Project Overview
- **Objective:** Predict the lift curve (CL vs. alpha) of aerofoils using a trained GRU-based neural network.
- **Input:** 14 CST parameters derived from aerofoil geometry files (.dat format).
- **Output:** 48 CL values and 48 angle-of-attack values spanning pre-stall, stall, and post-stall regions.
- **Data Source:** [UIUC Airfoil Database](https://m-selig.ae.illinois.edu/ads/coord_database.html) and [Airfoil Tools](http://airfoiltools.com/)

## 📁 Directory Structure
```bash
project/
├── curve_gen/          
│   ├── cg_scalers/            # Scalers for input and output data
│   ├── cg_model.pt            # Trained model
│   ├── cg_params.json         # Trained models
│   └── curve_gen.py           # Class for generating lift curves from aerofoils
├── data/
│   ├── aerofoil_data/         # Aerofoil geometry data
│   └── csv/                   # CSV files containing full dataset
├── exploration/               # Data exploration and analysis in Jupyter Notebooks
├── optuna_results/
│   ├── json/                  # JSON files containing optimisation results and names
│   ├── loss_func_results/     # Results for different loss functions
│   └── plot_analysis.py       # Plotting results script
├── scripts/                   # Python scripts
├── paper/
│   └── Final_Report.pdf       # Full project paper
├── src/
│   ├── data_.py               # Data preprocessing
│   ├── errors_.py             # Error metrics
│   ├── losses_.py             # Loss functions
│   ├── models_.py             # Neural network models
│   ├── plot_.py               # Plotting utilities
│   ├── train_eval_.py         # Training and evaluation logic
│   ├── tuning_.py             # Hyperparameter tuning utilities
│   ├── utils_.py              # Utility functions
│   └── models_.py             # Neural network models and instantiation function
├── tests/                     # Unit tests
├── main.py                    # Run inference using CurveGen
├── requirements.txt           # Environment dependencies
└── README.md                  # This file
```


# Using CurveGen to Predict a Curve
=====================================

The `CurveGen` class is the final outcome of the Aerofoil Lift Curve Prediction project. It provides a simple interface for generating predicted lift curves based on inputted aerofoil geometry.

## Creating an Instance of the CurveGen Class
---------------------------------------------

To use the `CurveGen` class, you'll need to create an instance of the class. This can be done by importing the `CurveGen` class and calling its constructor:

```python
from curve_gen import CurveGen

cg = CurveGen(k_params=None, dat_path=None, coords=None, aero_name=None)
```

The `CurveGen` class constructor takes four optional parameters:

**One of these parameters must be provided**

* `k_params`: a dictionary of Kulfan parameters
* `dat_path`: the path to a .dat file containing aerofoil geometry data
* `coords`: a list of coordinates representing the aerofoil geometry
* `aero_name`: the name of the aerofoil

## Generating a Predicted Lift Curve
--------------------------------------

Once you have created an instance of the `CurveGen` class, you can generate a predicted lift curve by calling the `plot_curve` method:

```python
curve_gen.plot_curve()
```

This will generate a plot of the predicted lift curve, showing the relationship between the lift coefficient and the angle of attack.

## Example Use Cases
--------------------

Here are a few example use cases for the `CurveGen` class:

### Using Kulfan Parameters

```python
from curve_gen import CurveGen

k_params = {...}  # define Kulfan parameters
curve_gen = CurveGen(k_params=k_params)
curve_gen.plot_curve()
```

### Using a .dat File

```python
from curve_gen import CurveGen

dat_path = 'path/to/aerofoil.dat'
curve_gen = CurveGen(dat_path=dat_path)
curve_gen.plot_curve()
```

### Using Coordinates

```python
from curve_gen import CurveGen

coords = [...]  # define coordinates
curve_gen = CurveGen(coords=coords)
curve_gen.plot_curve()
```

### Using Aerofoil Name

```python
from curve_gen import CurveGen

aero_name = 'aerofoil_name'
curve_gen = CurveGen(aero_name=aero_name)
curve_gen.plot_curve()
```

These example use cases demonstrate how to use the `CurveGen` class to generate predicted lift curves based on different types of input data.

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/bengawith/curve_gen.git
cd curve_gen
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Predict with a New Aerofoil
```bash
python main.py
```
This runs the `CurveGen` class on a sample aerofoil and plots the predicted lift curve.

## Disclaimer
Some exploratory scripts have been excluded for clarity and conciseness. However, the core logic for reproduction and evaluation has been fully included in this repository.

## Contact
For questions or issues, feel free to [contact me](mailto:enrbgawi@ljmu.ac.uk) for more information.
