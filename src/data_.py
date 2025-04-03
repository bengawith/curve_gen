import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
import difflib
import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)
from sklearn.model_selection import train_test_split


class DataPrep:
    def __init__(self, csv_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.csv_path = str(csv_path)
        self._load_data()

    def _load_data(self):
        # Check if the CSV file exists
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        # Assume that the file name encodes n_weights and n_outs as before
        file = os.path.splitext(os.path.basename(self.csv_path))[0]
        try:
            n_weights = int((int(file.split("KP")[0]) - 2) / 2)
            n_outs = int(file.split("_")[1].replace("CLA", ""))
        except (IndexError, ValueError):
            raise ValueError("Filename does not match expected format: '14KP_48CLA.csv'.")
    
        # Validate that necessary columns exist
        self.X_cols = [f"lower_weight_{i}" for i in range(n_weights)] + \
                      [f"upper_weight_{i}" for i in range(n_weights)] + \
                      ["TE_thickness", "leading_edge_weight"]
        self.y_cols = [f"CL_{i}" for i in range(n_outs)] + [f"alpha_{i}" for i in range(n_outs)]
        for col in self.X_cols + self.y_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column in dataset: {col}")
        
        # Optionally, store aerofoil names if present.
        if "aerofoil_name" in df.columns:
            self.aerofoil_names = df["aerofoil_name"].tolist()
        else:
            self.aerofoil_names = list(range(len(df)))
        
        # Scale features and labels (as before)
        self.X_scaler = StandardScaler()
        df[self.X_cols] = self.X_scaler.fit_transform(df[self.X_cols])
        self.X = df[self.X_cols].values
        self.y_scaler = MinMaxScaler()
        df[self.y_cols] = self.y_scaler.fit_transform(df[self.y_cols])
        self.y = df[self.y_cols].values
        
        # Split into train/test, and also split aerofoil names
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            self.X, self.y, self.aerofoil_names, test_size=0.20, random_state=42
        )

        # Convert to PyTorch tensors
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_train_tensor, self.y_train_tensor),
            batch_size=32, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.X_test_tensor, self.y_test_tensor),
            batch_size=32, shuffle=False)
    
        # Save the aerofoil names for train and test
        self.train_names = names_train
        self.test_names = names_test
    
        # For convenience, store input and output sizes
        for X_batch, y_batch in self.train_loader:
            self.input_size = X_batch.shape[1]
            self.output_size = y_batch.shape[1]
            break
    
    def get_data(self):
        return {
            'X_train': self.X_train_tensor,
            'X_test': self.X_test_tensor,
            'y_train': self.y_train_tensor,
            'y_test': self.y_test_tensor,
            'train_loader': self.train_loader,
            'test_loader': self.test_loader,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'train_names': self.train_names,
            'test_names': self.test_names
        }

class DataPrepFeatureBinning:
    """
    Data preprocessing class with feature-based clustering for aerofoil binning.
    The idea is to cluster aerofoil features using K-Means and augment the input 
    with continuous distance features to each centroid.
    """
    def __init__(self, csv_path, batch_size=32, test_size=0.2, n_bins=10, use_cluster_distances=True):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_bins = n_bins
        self.use_cluster_distances = use_cluster_distances
        self._load_data()

    def _cluster_features(self, feature_data):
        kmeans = KMeans(n_clusters=self.n_bins, random_state=42)
        clusters = kmeans.fit_predict(feature_data)
        centroids = kmeans.cluster_centers_
        # Compute Euclidean distances from each sample to each centroid
        distances = np.linalg.norm(feature_data[:, np.newaxis] - centroids, axis=2)
        print("Clustered features")
        return clusters, centroids, distances

    def _load_data(self):
        df = pd.read_csv(self.csv_path)

        # Store aerofoil names if available; otherwise use indices.
        if "aerofoil_name" in df.columns:
            self.aerofoil_names = df["aerofoil_name"].tolist()
        else:
            self.aerofoil_names = list(range(len(df)))

        # Define Kulfan parameter columns (assuming 6 lower and 6 upper weights)
        kulfan_columns = ([f"lower_weight_{i}" for i in range(6)] +
                          [f"upper_weight_{i}" for i in range(6)] +
                          ["TE_thickness", "leading_edge_weight"])
        feature_data = df[kulfan_columns].values

        # Cluster features using K-Means
        clusters, centroids, distances = self._cluster_features(feature_data)
        df["bin"] = clusters

        # Option to use one-hot encoding instead of distances:
        encoder = OneHotEncoder(sparse_output=False)
        encoded_bins = encoder.fit_transform(df[["bin"]])
        
        # Choose the feature augmentation:
        if self.use_cluster_distances:
            # Add distances as additional features
            features_combined = np.hstack([feature_data, distances])
        else:
            # Add one-hot encoded bin labels
            features_combined = np.hstack([feature_data, encoded_bins])
        
        self.X = features_combined

        # Labels: 48 lift coefficients and 48 angles of attack
        label_columns = [f"CL_{i}" for i in range(48)] + [f"alpha_{i}" for i in range(48)]
        self.y = df[label_columns].values

        # Scale features and labels
        self.X_scaler = StandardScaler()
        self.X = self.X_scaler.fit_transform(self.X)
        self.y_scaler = StandardScaler()
        self.y = self.y_scaler.fit_transform(self.y)

        # Train-test split: include aerofoil names in the split
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            self.X, self.y, self.aerofoil_names, test_size=self.test_size, random_state=42
        )

        # Convert to PyTorch tensors and create DataLoaders
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Save the aerofoil names for training and testing splits.
        self.train_names = names_train
        self.test_names = names_test

        # Store input and output sizes (from first training batch)
        for X_batch, y_batch in self.train_loader:
            self.input_size = int(X_batch.shape[1])
            self.output_size = int(y_batch.shape[1])
            break
            
        self.df = df

    def get_data(self):
        return {
            'df': self.df,
            'X_train': self.X_train_tensor,
            'X_test': self.X_test_tensor,
            'y_train': self.y_train_tensor,
            'y_test': self.y_test_tensor,
            'train_loader': self.train_loader,
            'test_loader': self.test_loader,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'train_names': self.train_names,
            'test_names': self.test_names
        }


class DataPrepDynamicBinning:
    """
    Data preprocessing class with dynamic binning based on aerofoil name similarity.
    This class groups aerofoils into bins based on name similarity and then augments the input
    with either the normalized cluster label or one-hot encoding.
    """
    def __init__(self, csv_path, batch_size=32, test_size=0.2, n_bins=15, use_cluster_label_feature=True):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_bins = n_bins
        self.use_cluster_label_feature = use_cluster_label_feature
        self._load_data()

    def _dynamic_bin_names(self, aerofoil_names):
        n = len(aerofoil_names)
        similarity_matrix = np.zeros((n, n))
        for i, name1 in enumerate(aerofoil_names):
            for j, name2 in enumerate(aerofoil_names):
                similarity_matrix[i, j] = difflib.SequenceMatcher(None, name1, name2).ratio()
        clustering = AgglomerativeClustering(n_clusters=self.n_bins, linkage="complete")
        clusters = clustering.fit_predict(1 - similarity_matrix)
        print(f"Created {self.n_bins} clusters based on aerofoil name similarity.")
        return clusters

    def _load_data(self):
        df = pd.read_csv(self.csv_path)

        # Store aerofoil names if available; otherwise, use indices.
        if "aerofoil_name" in df.columns:
            self.aerofoil_names = df["aerofoil_name"].tolist()
        else:
            self.aerofoil_names = list(range(len(df)))

        # Bin aerofoils based on their names using dynamic clustering.
        clusters = self._dynamic_bin_names(self.aerofoil_names)
        df["bin"] = clusters

        # One-hot encode bin labels as an option.
        encoder = OneHotEncoder(sparse_output=False)
        encoded_bins = encoder.fit_transform(df[["bin"]])

        # Define Kulfan parameter columns (assuming 6 lower and 6 upper weights).
        kulfan_columns = ([f"lower_weight_{i}" for i in range(6)] +
                          [f"upper_weight_{i}" for i in range(6)] +
                          ["TE_thickness", "leading_edge_weight"])
        kulfan_params = df[kulfan_columns].values

        # Augment features with either normalized cluster label or one-hot encoded bins.
        if self.use_cluster_label_feature:
            bin_feature = df["bin"].values.reshape(-1, 1).astype(float)
            bin_feature = (bin_feature - bin_feature.min()) / (bin_feature.max() - bin_feature.min())
            features_combined = np.hstack([kulfan_params, bin_feature])
        else:
            features_combined = np.hstack([kulfan_params, encoded_bins])

        self.X = features_combined

        # Labels: 48 lift coefficients and 48 angles of attack.
        label_columns = [f"CL_{i}" for i in range(48)] + [f"alpha_{i}" for i in range(48)]
        self.y = df[label_columns].values

        # Scale features and labels.
        self.X_scaler = StandardScaler()
        self.X = self.X_scaler.fit_transform(self.X)
        self.y_scaler = StandardScaler()
        self.y = self.y_scaler.fit_transform(self.y)

        # Split into training and test sets, including the aerofoil names.
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            self.X, self.y, self.aerofoil_names, test_size=self.test_size, random_state=42
        )

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        self.train_names = names_train
        self.test_names = names_test

        for X_batch, y_batch in self.train_loader:
            self.input_size = int(X_batch.shape[1])
            self.output_size = int(y_batch.shape[1])
            break

        self.df = df

    def get_data(self):
        return {
            'df': self.df,
            'X_train': self.X_train_tensor,
            'X_test': self.X_test_tensor,
            'y_train': self.y_train_tensor,
            'y_test': self.y_test_tensor,
            'train_loader': self.train_loader,
            'test_loader': self.test_loader,
            'X_scaler': self.X_scaler,
            'y_scaler': self.y_scaler,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'train_names': self.train_names,
            'test_names': self.test_names
        }
