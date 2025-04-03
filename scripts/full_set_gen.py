import os
import numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
import pandas as pd
from multiprocessing import Pool
import requests
from io import StringIO
import sys


dat_directory = "./data/aerofoil_data"

def read_airfoil_coordinates(file_path):
    """Reads airfoil coordinates from a .dat file."""
    coordinates = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            try:
                x, y = map(float, line.split())
                coordinates.append([x, y])
            except ValueError:
                continue
    return np.array(coordinates)

def process_airfoil(filename, weights, n_points, method):
    """Processes a single airfoil file and returns the data row."""
    aerofoil_name = filename.split('.')[0]
    file_path = os.path.join(dat_directory, filename)
    coordinates = read_airfoil_coordinates(file_path)

    try:
        # Get Kulfan parameters
        kulfan_params = get_kulfan_parameters(coordinates=coordinates, n_weights_per_side=int(weights), method=method)
        lower_weights = kulfan_params['lower_weights']
        upper_weights = kulfan_params['upper_weights']
        te_thickness = kulfan_params['TE_thickness']
        leading_edge_weight = kulfan_params['leading_edge_weight']

        # Prepare the row to be added to the CSV
        row = [aerofoil_name] + lower_weights.tolist() + upper_weights.tolist() + [te_thickness, leading_edge_weight] + [""] * int(n_points) + [""] * int(n_points)
        return row

    except Exception as e:
        print(f"Error processing {aerofoil_name}: {e}")
        return None

def process_airfoils_and_save_to_csv(weights, n_points, method):
    """Processes all airfoils in the directory and saves Kulfan Parameters to a CSV."""
    headers = ["aerofoil_name"] + [f"lower_weight_{i}" for i in range(int(weights))] + [f"upper_weight_{i}" for i in range(int(weights))] + ["TE_thickness", "leading_edge_weight"] + [f"CL_{i}" for i in range(int(n_points))] + [f"alpha_{i}" for i in range(int(n_points))]

    with Pool() as pool:
        rows = pool.starmap(process_airfoil, [(f, weights, n_points, method) for f in os.listdir(dat_directory) if f.endswith('.dat')])

    rows = [row for row in rows if row is not None]
    df = pd.DataFrame(rows, columns=headers)
    df_sorted = df.sort_values(by=df.columns[0])
    csv_filename = f"./data/csv/{(weights*2)+2}KP_{n_points}CLA_{method}.csv"
    df_sorted.to_csv(csv_filename, index=False)
    print(f"The CSV file has been sorted by the first column and saved as {csv_filename}.")

def get_cl_values(aerofoil_name, n_points):
    """Fetches CL and alpha values for a given aerofoil."""
    base_url = "http://airfoiltools.com/polar/csv?polar=xf-{aerofoil}-il-1000000"
    url = base_url.format(aerofoil=str(aerofoil_name).lower())

    try:
        response = requests.head(url)
        if response.status_code == 404:
            print(f"Data not found for {aerofoil_name} at {url}.")
            return [None] * (2 * n_points)
        
        response = requests.get(url)
        response.raise_for_status()

        lines = response.text.splitlines()
        header_index = next(i for i, line in enumerate(lines) if all(col in line.lower() for col in ['alpha', 'cl', 'cd', 'cdp', 'cm', 'top_xtr', 'bot_xtr']))
        data_text = "\n".join(lines[header_index:])
        aerofoil_df = pd.read_csv(StringIO(data_text))

        aerofoil_df.columns = [col.strip().lower() for col in aerofoil_df.columns]
        alpha_col = next(col for col in aerofoil_df.columns if 'alpha' in col)
        cl_col = next(col for col in aerofoil_df.columns if 'cl' in col)

        aerofoil_df = aerofoil_df.sort_values(by=alpha_col).reset_index(drop=True)
        indices = np.linspace(0, len(aerofoil_df) - 1, n_points, dtype=int)
        selected_data = aerofoil_df.iloc[indices]

        cl_values = selected_data[cl_col].tolist()
        alpha_values = selected_data[alpha_col].tolist()
        return cl_values + alpha_values

    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve data for {aerofoil_name} at {url}: {e}")
        return [None] * (2 * n_points)

def update_aerofoil_data(row, n_points):
    """Updates a row with CL and alpha values."""
    aerofoil_name = row['aerofoil_name']
    cl_cols = [f"CL_{i}" for i in range(n_points)]
    alpha_cols = [f"alpha_{i}" for i in range(n_points)]

    if not row[cl_cols + alpha_cols].isnull().any():
        return row

    cl_and_alpha_values = get_cl_values(aerofoil_name, n_points)
    for i, col in enumerate(cl_cols + alpha_cols):
        row[col] = cl_and_alpha_values[i]
    
    return row

def full_dataset():
    weights = 6
    n_points = 48
    method = 'least_squares'

    process_airfoils_and_save_to_csv(weights, n_points, method)
    
    csv_filename = f"./data/csv/{(weights*2)+2}KP_{n_points}CLA_{method}.csv"
    df = pd.read_csv(csv_filename)
    
    with Pool() as pool:
        updated_rows = pool.starmap(update_aerofoil_data, [(row, n_points) for _, row in df.iterrows()])

    updated_df = pd.DataFrame(updated_rows, columns=df.columns)
    updated_df.rename(columns=lambda x: x.strip(), inplace=True)
    updated_df.replace("", pd.NA, inplace=True)
    updated_df.dropna(axis=0, how='any', inplace=True)
    
    updated_df.to_csv(csv_filename, index=False)
    
    print(f"Updated CSV saved as {csv_filename} and NaN dropped.")

if __name__ == "__main__":
    full_dataset()