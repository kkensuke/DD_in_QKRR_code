import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from scipy.optimize import bisect
# from scipy.special import erf
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def load_fashion_mnist(N_QUBITS, verbose=False, seed=42):
    """Read Fashion-MNIST with caching"""
    cache_path = Path.home()/".cache"/"DD_QKRR"/"fashion_mnist.npz"
    
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X, y = data["X"], data["y"]
    else:
        print("Downloading...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
        np.savez_compressed(cache_path, X=X, y=y)
        print(f"Download complete: {cache_path}")
    
    X = X.astype(np.float32)
    y = y.astype(int)

    if verbose:
        print(f"Shape: X={X.shape}, y={y.shape}")

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=45000, test_size=25000, random_state=seed)

    # standardize data -- make mean 0 and std 1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # choose only two classes 0 and 1 for now
    train_filter = np.where((y_train == 0) | (y_train == 1))
    test_filter = np.where((y_test == 0) | (y_test == 1))

    # reduced training and test data with only 2 classes
    X_train_filtered = X_train[train_filter]
    y_train_filtered = y_train[train_filter]
    X_test_filtered = X_test[test_filter]
    y_test_filtered = y_test[test_filter]

    if verbose:
        print(f"After filtering two classes: train = {X_train_filtered.shape[0]}, test = {X_test_filtered.shape[0]}")

    # Dimensionality Reduction with PCA
    PCA_COMPONENTS = 2 * N_QUBITS

    pca = PCA(n_components=PCA_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_filtered)
    X_test_pca = pca.transform(X_test_filtered)

    if verbose:
        print(f"Shape after PCA: X_train_pca = {X_train_pca.shape}, X_test_pca = {X_test_pca.shape}")

    # # One-Hot Encoding
    # y_train_reshaped = y_train_filtered.reshape(-1, 1)
    # y_test_reshaped = y_test_filtered.reshape(-1, 1)

    # OHE = OneHotEncoder()
    # OHE.fit(y_train_reshaped)
    # y_train_ohe = OHE.transform(y_train_reshaped).toarray()
    # y_test_ohe = OHE.transform(y_test_reshaped).toarray()

    # if verbose:
    #     print(f"One-hot encoded shape: y_train_ohe = {y_train_ohe.shape}, y_test_ohe = {y_test_ohe.shape}")

    # Select Test Data Subset
    X_train_final = X_train_pca
    X_test_final = X_test_pca
    # y_train_final = y_train_ohe
    # y_test_final = y_test_ohe
    y_train_final = y_train_filtered
    y_test_final = y_test_filtered

    if verbose:
        print(f"Final data shape: train={X_train_final.shape}, test={X_test_final.shape}")
    
    return X_train_final, y_train_final, X_test_final, y_test_final


class SyntheticDataset:
    def __init__(self,
                 N_QUBITS: int = 3,
                 train_size: int = 20000,
                 test_size: int = 10000,
                 noise_sigma: float = 0.3,
                 seed: int = 42
                 ):
        self.N_QUBITS = N_QUBITS
        self.train_size = train_size
        self.test_size = test_size
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        self.x_dim = 2 * N_QUBITS
    
    def generate_true_params(self) -> np.ndarray:
        params = self.rng.uniform(0, 2 * np.pi, size=(self.x_dim,))
        return params
    
    @staticmethod
    def true_function(X, true_params) -> np.ndarray:
        # X shape: (sample_size, x_dim)
        sin_X = np.sin(X)
        inner = np.dot(sin_X, true_params)
        return inner
    
    def generate_dataset(self, true_params=None):
        self.true_params = true_params if true_params is not None else self.generate_true_params()
        X = self.rng.normal(0, 1, size=(self.train_size + self.test_size, self.x_dim))
        y = self.true_function(X, self.true_params) + self.noise_sigma * self.rng.normal(0, 1, size=(self.train_size + self.test_size,))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=self.train_size, test_size=self.test_size)
        return X_train, y_train, X_test, y_test


# Under development
class SyntheticQuantumDataset:
    def __init__(self,
                 N_QUBITS: int = 3,
                 train_size: int = 20000,
                 test_size: int = 10000,
                 noise_sigma: float = 0.3,
                 seed: int = 42
                 ):
        self.N_QUBITS = N_QUBITS
        self.train_size = train_size
        self.test_size = test_size
        self.noise_sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
    
    def generate_true_params(self) -> np.ndarray:
        dim = 4 ** self.N_QUBITS
        params = self.rng.normal(0, 1, size=(dim,))
        return params


class QuantumKernel:
    def __init__(self, N_QUBITS: int = 3, TPA: bool = False):
        self.N_QUBITS = N_QUBITS
        self.TPA = TPA
        self.state_vector = self.make_state_vector_function()

    def _internal_circuit(self, x):
        """Internal quantum circuit for feature map
        When more features than qubits, features are mapped to qubits as x[i % self.N_QUBITS]
        Args:
            x (array): Input data point
        """
        n_layers = self.N_QUBITS
        for _ in range(n_layers):
            # RX rotations and RZ rotations
            for i in range(len(x)):
                qml.RX(x[i], wires=i % self.N_QUBITS)
                qml.RZ(x[i], wires=i % self.N_QUBITS)
            
            # Entanglement
            for i in range(self.N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
    
    def _internal_circuit_TPA(self, x):
        n_layers = self.N_QUBITS
        for _ in range(n_layers):
            for i in range(len(x)):
                qml.RX(x[i], wires=i % self.N_QUBITS)

    def make_state_vector_function(self):
        """Create a state vector function for given number of qubits"""
        dev = qml.device("lightning.qubit", wires=self.N_QUBITS)
        
        @qml.qnode(dev)
        def state_vector(x1):
            """Return state vector"""
            if self.TPA:
                self._internal_circuit_TPA(x1)
            else:
                self._internal_circuit(x1)
            return qml.state()
        
        return state_vector

    def kernel_matrix(self, X1, X2, same: bool = False) -> np.ndarray:
        """Compute quantum kernel matrix"""
        if same:
            QX = np.array([self.state_vector(x) for x in X1], dtype=complex).T # 2**N_QUBITS x X1.shape[0]
            inner = QX.conj().T @ QX # X1.shape[0] x X1.shape[0]
        else:
            QX1 = np.array([self.state_vector(x) for x in X1], dtype=complex).T # 2**N_QUBITS x X1.shape[0]
            QX2 = np.array([self.state_vector(x) for x in X2], dtype=complex).T # 2**N_QUBITS x X2.shape[0]
            inner = QX1.conj().T @ QX2 # X1.shape[0] x X2.shape[0]
        
        K = np.abs(inner) ** 2
        return K.astype(np.float64) # sample_size1 x sample_size2

    def Sigma(self, X1, X2):
        """Quantum kernel function"""
        QX1 = np.array([np.kron(self.state_vector(x).conj(), self.state_vector(x)) for x in X1], dtype=complex).T # 4**N_QUBITS x X1.shape[0]
        QX2 = np.array([np.kron(self.state_vector(x).conj(), self.state_vector(x)) for x in X2], dtype=complex).T # 4**N_QUBITS x X2.shape[0]
        Sigma_matrix = QX1 @ QX2.conj().T
        return Sigma_matrix.astype(np.complex128) # 4**N_QUBITS x 4**N_QUBITS


# This class computes the numerical Test error for a list of regularization coefficients $\lambda$.
class NumericalTestRisk:
    def __init__(self,
                 N_QUBITS: int = 3,
                 LAMBDA_LIST: list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                 REPS: int = 3,
                 TPA: bool = False
                 ):
        self.N_QUBITS = N_QUBITS
        self.LAMBDA_LIST = LAMBDA_LIST
        self.REPS = REPS
        self.TPA = TPA
        self.quantum_kernel = QuantumKernel(N_QUBITS, TPA)
        
        self.n_train_list = self.get_n_train_list()
        self.max_train = max(self.n_train_list)
        print(f"Training sample sizes: {self.n_train_list}")

    def get_n_train_list(self):
        """Get explicit list of training sample sizes based on the number of qubits"""
        # For HEA, the kernel matrix has rank 4^N_QUBITS, so we set the training range around 4^N_QUBITS.
        hea_lists = {
            1: [1, 2, 3, 4, 5, 6],
            2: [10, 12, 14, 16, 18, 20],
            3: [40, 48, 56, 64, 72, 80],
            4: [160, 192, 224, 256, 288, 320],
            5: [640, 768, 896, 1024, 1152, 1280]
        }

        # For TPA, the kernel matrix has rank 3^N_QUBITS, so we set the training range around 3^N_QUBITS. The list is obtained by multiplying the HEA list by (3/4)^n and rounding to the nearest integer.
        tpa_lists = {
            1: [1, 2, 3, 4, 5, 6],
            2: [6, 7, 8, 9, 10, 11],
            3: [17, 20, 24, 27, 30, 34],
            4: [51, 61, 71, 81, 91, 101],
            5: [152, 182, 213, 243, 273, 304]
        }
        
        if self.TPA:
            return tpa_lists.get(self.N_QUBITS)
        else:
            return hea_lists.get(self.N_QUBITS)

    def validate_data_availability(self, x_train):
        required_samples = self.REPS * self.max_train
        available_samples = x_train.shape[0]
        if available_samples < required_samples:
            raise ValueError(
                f"Not enough training samples! "
                f"Required: {required_samples}, Available: {available_samples}"
            )
        else:
            print(f"Training data available: {available_samples} samples, required: {required_samples} samples")

    def train_and_evaluate(self, x_train, x_test, y_train, y_test, lam):
        """Train quantum kernel ridge regression model and evaluate predictions.
            Uses analytical solution: α = (K + n_train λI)^(-1)y,
            where K is the precomputed kernel matrix K[i,j] = |⟨ψ(x_i)|ψ(x_j)⟩|²
            and ψ(x) is the quantum state vector.
        Args:
            K_train: Training kernel matrix (n_train × n_train)
            K_test: Test kernel matrix (n_test × n_train)
            y_train: Training labels (one-hot encoded)
            y_test: Test labels (one-hot encoded)
        
        Returns:
            model: Fitted KernelRidge model
            mse_test: Mean squared error on test set
            mse_train: Mean squared error on training set
        """
        K_train = self.quantum_kernel.kernel_matrix(x_train, x_train, same=True)
        K_test = self.quantum_kernel.kernel_matrix(x_test, x_train)
        
        train_size = K_train.shape[0]
        # dual_estimator = np.linalg.inv(K_train + train_size * lam * np.eye(train_size)) @ y_train
        # `np.linalg.solve` is several times faster than `np.linalg.inv`
        dual_estimator = np.linalg.solve(K_train + train_size * lam * np.eye(train_size), y_train)
        
        predictions_train = K_train @ dual_estimator
        predictions_test = K_test @ dual_estimator
        
        mse_train = np.mean((y_train - predictions_train) ** 2)
        mse_test = np.mean((y_test - predictions_test) ** 2)
        
        model = None
        
        return model, mse_test, mse_train
    
    def GCV_risk(self, K_train_empirical, y_train):
        # GCV risk estimator = 1/N * y^T (K/N + λI)^(-2) y / (1/N Tr[(K/N + λI)^(-1)])^2
        N = K_train_empirical.shape[0]
        GCV_risk_list = []
        for i, lam in enumerate(self.LAMBDA_LIST):
            inv = np.linalg.inv(K_train_empirical / N + lam * np.eye(N))
            GCV_risk_lam = y_train.T @ inv @ inv @ y_train / N
            trace_term = np.trace(inv) / N
            GCV_risk_lam /= trace_term ** 2
            GCV_risk_list.append(GCV_risk_lam)
        
        return GCV_risk_list
    
    def numerical_test_risk_lambda(self, X_train, y_train, X_test, y_test):
        self.validate_data_availability(X_train)

        # Dictionary to save results (self.LAMBDA_LIST x self.REPS x training_sizes)
        mse_train_all = {lam: [] for lam in self.LAMBDA_LIST}
        mse_test_all = {lam: [] for lam in self.LAMBDA_LIST}

        for lambda_val in self.LAMBDA_LIST:
            print(f"\nλ = {lambda_val}")
            
            for rep in range(self.REPS):
                print(f"  Rep {rep+1}/{self.REPS}", end='\r')
                mse_train_lambda = []
                mse_test_lambda = []
                
                # Use different samples in each iteration
                start_idx = rep * self.max_train
                end_idx = (rep + 1) * self.max_train
                X_train_max = X_train[start_idx:end_idx]
                y_train_max = y_train[start_idx:end_idx]
                
                for n_train in self.n_train_list:
                    x_train_subset = X_train_max[:n_train]
                    y_train_subset = y_train_max[:n_train]
                    
                    model, mse_test, mse_train = self.train_and_evaluate(
                        x_train_subset, X_test, y_train_subset, y_test, lambda_val
                    )
                    
                    mse_train_lambda.append(mse_train)
                    mse_test_lambda.append(mse_test)
                
                mse_train_all[lambda_val].append(mse_train_lambda)
                mse_test_all[lambda_val].append(mse_test_lambda)
            print("  Completed!")

        self.mse_train_all_mean = {lam: np.mean(mse_train_all[lam], axis=0) for lam in self.LAMBDA_LIST}
        self.mse_train_all_std = {lam: np.std(mse_train_all[lam], axis=0) for lam in self.LAMBDA_LIST}
        self.mse_test_all_mean = {lam: np.mean(mse_test_all[lam], axis=0) for lam in self.LAMBDA_LIST}
        self.mse_test_all_std = {lam: np.std(mse_test_all[lam], axis=0) for lam in self.LAMBDA_LIST}
        # print(np.array(mse_train_all[1e-8]).shape == (self.REPS, len(self.n_train_list)))
        # print(np.mean(np.array(mse_train_all[1e-8]), axis=0))
    
    def numerical_test_risk_lambda_optimized(self, X_train, y_train, X_test, y_test):
        """Optimized version of `numerical_test_risk_lambda` to reuse precomputed kernel matrices."""
        self.validate_data_availability(X_train)
        
        # Dictionary to save results (self.REPS x training_sizes x self.LAMBDA_LIST)
        mse_train_all = []
        mse_test_all = []
        GCV_risk_all = []
        
        for rep in range(self.REPS):
            print(f"  Rep {rep+1}/{self.REPS}", end='\r')
            mse_train_rep = []
            mse_test_rep = []
            
            # Use different samples in each iteration
            start_idx = rep * self.max_train
            end_idx = (rep + 1) * self.max_train
            X_train_max = X_train[start_idx:end_idx]
            y_train_max = y_train[start_idx:end_idx]
            
            K_train_max = self.quantum_kernel.kernel_matrix(X_train_max, X_train_max, same=True)
            K_test_max = self.quantum_kernel.kernel_matrix(X_test, X_train_max)
            
            for n_train in self.n_train_list:
                mse_train_n_train = []
                mse_test_n_train = []
                y_train_subset = y_train_max[:n_train]
                
                K_train_subset = K_train_max[:n_train, :n_train]
                K_test_subset = K_test_max[:, :n_train]
                
                train_size = K_train_subset.shape[0]
                
                s, U = np.linalg.eigh(K_train_subset)
                Uy = U.T @ y_train_subset
                for lambda_val in self.LAMBDA_LIST:
                    # dual_estimator = np.linalg.solve(K_train_subset + train_size * lambda_val * np.eye(train_size), y_train_subset)
                    dual_estimator = (U * (1.0 / (s + train_size * lambda_val))) @ Uy
                    
                    predictions_train = K_train_subset @ dual_estimator
                    mse_train = np.mean((y_train_subset - predictions_train) ** 2)
                    predictions_test = K_test_subset @ dual_estimator
                    mse_test = np.mean((y_test - predictions_test) ** 2)
                    
                    mse_train_n_train.append(mse_train)
                    mse_test_n_train.append(mse_test)
                mse_train_rep.append(mse_train_n_train)
                mse_test_rep.append(mse_test_n_train)
                
                if rep == 0:
                    gcv_risk = self.GCV_risk(K_train_subset, y_train_subset)
                    GCV_risk_all.append(gcv_risk)
            
            mse_train_all.append(mse_train_rep)
            mse_test_all.append(mse_test_rep)
        print("  Completed!")
        
        self.GCV_risk_list = np.array(GCV_risk_all).T

        self.mse_train_all_mean = dict(zip(self.LAMBDA_LIST, np.mean(mse_train_all, axis=0).T))
        self.mse_train_all_std = dict(zip(self.LAMBDA_LIST, np.std(mse_train_all, axis=0).T))
        self.mse_test_all_mean = dict(zip(self.LAMBDA_LIST, np.mean(mse_test_all, axis=0).T))
        self.mse_test_all_std = dict(zip(self.LAMBDA_LIST, np.std(mse_test_all, axis=0).T))
    
    def plot_mse_train(self):
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.LAMBDA_LIST)))
        # Dynamically set ratio base depending on the circuit type
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
            xlabel_str = f'$3^{{{self.N_QUBITS}}}/N_{{tr}}$'
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)
            xlabel_str = f'$4^{{{self.N_QUBITS}}}/N_{{tr}}$'
            
        for i, lambda_val in enumerate(self.LAMBDA_LIST):
            plt.semilogy(
                ratio_list, 
                self.mse_train_all_mean[lambda_val], 
                marker='o', 
                alpha=0.8, 
                label=f'λ = {lambda_val:.0e}', 
                linestyle='none', 
                color=colors[i]
            )

        plt.xlabel(xlabel_str, fontsize=12)
        plt.title(f'Numerical Train Error ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend(title='Regularization λ')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_mse_test(self):
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.LAMBDA_LIST)))
        # Dynamically set ratio base depending on the circuit type
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
            xlabel_str = f'$3^{{{self.N_QUBITS}}}/N_{{tr}}$'
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)
            xlabel_str = f'$4^{{{self.N_QUBITS}}}/N_{{tr}}$'
            
        for i, lambda_val in enumerate(self.LAMBDA_LIST):
            plt.semilogy(
                ratio_list, 
                self.mse_test_all_mean[lambda_val], 
                marker='o', 
                alpha=0.8, 
                label=f'λ = {lambda_val:.0e}', 
                linestyle='none', 
                color=colors[i]
            )

        plt.xlabel(xlabel_str, fontsize=12)
        plt.title(f'Numerical Test Error ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend(title='Regularization λ')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_mse_test_std(self):
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.LAMBDA_LIST)))
        # Dynamically set ratio base depending on the circuit type
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
            xlabel_str = f'$3^{{{self.N_QUBITS}}}/N_{{tr}}$'
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)
            xlabel_str = f'$4^{{{self.N_QUBITS}}}/N_{{tr}}$'
            
        for i, lambda_val in enumerate(self.LAMBDA_LIST):
            plt.errorbar(
                ratio_list, 
                self.mse_test_all_mean[lambda_val], 
                yerr=self.mse_test_all_std[lambda_val], 
                marker='o', 
                capsize=3,
                alpha=0.8, 
                label=f'λ = {lambda_val:.0e}', 
                linestyle='none', 
                color=colors[i]
            )

        plt.yscale('log')
        plt.xlabel(xlabel_str, fontsize=12)
        plt.title(f'Numerical Test Error ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend(title='Regularization λ')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_mse_xaxis_lambda(self):
        # Plot MSE with x-axis as lambda
        # Set `self.n_train_list` by yourself before calling `numerical_test_risk_lambda_optimized`
        # e.g., self.n_train_list = [0.5 * 4**N_QUBITS, 1.0 * 4**N_QUBITS, 2.0 * 4**N_QUBITS]
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.n_train_list)))
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)
        mse_test_all_mean_list = [self.mse_test_all_mean[lam] for lam in self.LAMBDA_LIST]
        mse_test_all_mean_list = np.array(mse_test_all_mean_list).T  # shape: (len(self.n_train_list), len(self.LAMBDA_LIST))

        for i, ratio in enumerate(ratio_list):
            plt.semilogy(self.LAMBDA_LIST, mse_test_all_mean_list[i], marker='o', alpha=0.7, label=f'γ = {ratio:.2f}', linestyle='none', color=colors[i], markersize=5)
        plt.xscale('log')
        plt.xlabel('$\lambda$', fontsize=12)
        plt.title(f'Numerical Test Error vs λ ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_GCV_risk(self):
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.LAMBDA_LIST)))
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)
        
        for i, lambda_val in enumerate(self.LAMBDA_LIST):
            plt.semilogy(ratio_list, self.GCV_risk_list[i], marker='o', alpha=0.8, label=f'λ = {lambda_val:.0e}', linestyle='none', color=colors[i])

        plt.xlabel(f'$4^{{{self.N_QUBITS}}}/N_{{tr}}$', fontsize=12)
        # plt.ylim(1e1, 1e6)
        plt.title(f'GCV Risk Estimator ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend(title='Regularization λ')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_results(self, path: str) -> None:
        """
        Save variables into a single pickle file atomically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "LAMBDA_LIST": self.LAMBDA_LIST,
            "n_train_list": self.n_train_list,
            "GCV_risk_list": self.GCV_risk_list,
            "mse_test_all_mean": self.mse_test_all_mean,
            "mse_test_all_std": self.mse_test_all_std,
            "mse_train_all_mean": self.mse_train_all_mean,
            "mse_train_all_std": self.mse_train_all_std
        }
        # write to temp then replace to avoid corrupting existing file
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    
    def load_results(self, path: str) -> dict:
        """
        Usage:
            instance = NumericalTestRisk()
            instance.load_results("results/asdf.pkl")
            instance.plot_numerical()
        """
        with open(path, "rb") as f:
            self.loaded = pickle.load(f)

        self.LAMBDA_LIST = self.loaded["LAMBDA_LIST"]
        self.n_train_list = self.loaded["n_train_list"]
        self.GCV_risk_list = self.loaded["GCV_risk_list"]
        self.mse_test_all_mean = self.loaded["mse_test_all_mean"]
        self.mse_test_all_std = self.loaded["mse_test_all_std"]
        self.mse_train_all_mean = self.loaded["mse_train_all_mean"]
        self.mse_train_all_std = self.loaded["mse_train_all_std"]


## Functions (`solve_self_consistent_eq`) to solve self-consistent equations: $\frac{1}{n}\operatorname{Tr}[\Sigma(\Sigma + \kappa_\lambda I)^{-1}] = 1 - \frac{\lambda}{\kappa_\lambda}$.
# If we use the cost function has the following form, which does not include normalization by the number of training samples $n$:
# $$
# L(\beta) = \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda ||\beta||^2,
# $$
# then the self-consistent equation to determine $\kappa_\lambda$ is given by:
# $$
# \operatorname{Tr}[\Sigma(\Sigma + \kappa_\lambda I)^{-1}] = n - \frac{\lambda}{\kappa_\lambda}
# $$
# This self-consistent equation is from the paper: [Dimension-free deterministic equivalents and scaling laws for random feature regression](https://arxiv.org/abs/2405.15699). This little difference is due to the difference in the definition of cost function.
# Since this self-consistent equation can be transformed into a following form:
# $$
# \frac{1}{n}\operatorname{Tr}[\Sigma(\Sigma + \kappa_\lambda I)^{-1}] = 1 - \frac{\lambda/n}{\kappa_\lambda}
# $$
# The solution $\kappa_\lambda$ of this equation is the same as the solution of the original self-consistent equation with $\lambda$ replaced by $\lambda/n$.

def solve_self_consistent_eq(Sigma, n_train, lam):
    if lam == 0:
        return 0.0 if np.all(Sigma > 0) else np.nan
    
    def f_of_kappa(kappa):
        return np.sum(Sigma / (Sigma + kappa)) / n_train - (1.0 - lam / kappa)
    
    return bisect(f_of_kappa, 1e-12, 1e6, xtol=1e-10)

# This class computes the theoretical test risk for a list of regularization coefficients $\lambda$.
class TheoreticalTestRisk:
    def __init__(self,
                 N_QUBITS: int = 3,
                 LAMBDA_LIST: list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
                 noise_sigma: float = 0.3,
                 TPA: bool = False
                 ):
        self.N_QUBITS = N_QUBITS
        self.LAMBDA_LIST = LAMBDA_LIST
        self.noise_sigma: float = noise_sigma
        self.TPA = TPA
        self.quantum_kernel = QuantumKernel(N_QUBITS, TPA)
        
        self.n_train_list = np.arange(*self.get_training_range())
    
    def get_training_range(self):
        """Get training sample range based on number of qubits"""
        if self.TPA:
            # For TPA, the kernel matrix has rank 3^N_QUBITS, so we set the training range around 3^N_QUBITS.
            ranges = {
                1: (1, 7, 1),
                2: (5, 11, 1),
                3: (17, 36, 2),
                4: (3**self.N_QUBITS - 30, 3**self.N_QUBITS + 20, 5),
                5: (3**self.N_QUBITS - 93, 3**self.N_QUBITS + 70, 3)
            }
        else:
            # For HEA, the kernel matrix has rank 4^N_QUBITS, so we set the training range around 4^N_QUBITS.
            ranges = {
                1: (1, 7, 1),
                2: (4**self.N_QUBITS - 7, 4**self.N_QUBITS + 5, 1),
                3: (4**self.N_QUBITS - 24, 4**self.N_QUBITS + 19, 2),
                4: (4**self.N_QUBITS - 90, 4**self.N_QUBITS + 70, 5),
                5: (4**self.N_QUBITS - 390, 4**self.N_QUBITS + 270, 10)
            }
        result = ranges.get(self.N_QUBITS)
        return result
    
    # Empirically Estimate $\Sigma$ and $\beta_*$ from the kernel and substitute into theoretical expression of Deterministic Equivalent.
    # Referred to the algorithm 1 in this paper: [Dimension-free deterministic equivalents and scaling laws for random feature regression](https://arxiv.org/abs/2405.15699).
    def K_eigen_and_beta(self, X_test, y_test):
        """Estimate Sigma eigenvalues and beta from test data.
        Args:
            X_test: Test data inputs
            y_test: Test data outputs (noise-free true function values preferred)
        Returns:
            K_eigenvalues: Eigenvalues of empirical kernel matrix on test data
            beta: Projection of true function values onto eigenvectors of empirical kernel matrix
        """
        K_empirical = self.quantum_kernel.kernel_matrix(X_test, X_test, same=True)
        n_test = y_test.shape[0]
        
        K_eigenvalues, K_eigenvectors = np.linalg.eigh(K_empirical/n_test)
        # K_empirical @ K_eigenvectors.T[0] equals K_eigenvalues[0] * K_eigenvectors.T[0]
        
        beta = K_eigenvectors.T @ y_test / np.sqrt(n_test)
        return K_eigenvalues, beta
    
    def theoretical_test_risk(self, K_eigenvalues, beta, n_train, lam):
        kappa = solve_self_consistent_eq(K_eigenvalues, n_train, lam)
        delta = np.sum(
            (K_eigenvalues + kappa)**(-2) * K_eigenvalues**2
        ) / n_train
        
        Bias = beta.T @ ((K_eigenvalues + kappa)**(-2) * beta) * kappa**2 / (1 - delta)
        Variance = self.noise_sigma**2 * delta / (1 - delta)
        
        R_theoretical = Bias + Variance + self.noise_sigma**2
        Train_theoretical = R_theoretical * lam**2 / kappa**2
        return delta, Bias, Variance, R_theoretical, Train_theoretical
    
    def theoretical_test_risk_lambda(self, X_test, y_test):
        self.K_eigenvalues, beta = self.K_eigen_and_beta(X_test, y_test)

        effective_DOF_list = []
        Bias_list = []
        Variance_list = []
        R_theoretical_list = []
        Train_theoretical_list = []
        for i, lam in enumerate(self.LAMBDA_LIST):
            print(f"\nλ = {lam}")
            effective_DOF_list_lam = []
            Bias_list_lam = []
            Variance_list_lam = []
            R_theoretical_list_lam = []
            Train_theoretical_list_lam = []
            
            for n_train in self.n_train_list:
                print(f"  n_train = {n_train}", end='\r')
                delta, Bias, Variance, R_theoretical, Train_theoretical = self.theoretical_test_risk(self.K_eigenvalues, beta, n_train, lam)
                
                effective_DOF_list_lam.append(delta)
                Bias_list_lam.append(Bias)
                Variance_list_lam.append(Variance)
                R_theoretical_list_lam.append(R_theoretical)
                Train_theoretical_list_lam.append(Train_theoretical)
                
            effective_DOF_list.append(effective_DOF_list_lam)
            Bias_list.append(Bias_list_lam)
            Variance_list.append(Variance_list_lam)
            R_theoretical_list.append(R_theoretical_list_lam)
            Train_theoretical_list.append(Train_theoretical_list_lam)
            
        self.effective_DOF_list = np.array(effective_DOF_list)
        self.Bias_list = np.array(Bias_list)
        self.Variance_list = np.array(Variance_list)
        self.R_theoretical_list = np.array(R_theoretical_list)
        self.Train_theoretical_list = np.array(Train_theoretical_list)
    
    def save_results(self, path: str) -> None:
        """
        Save variables into a single pickle file atomically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "N_QUBITS": self.N_QUBITS,
            "LAMBDA_LIST": self.LAMBDA_LIST,
            "n_train_list": self.n_train_list,
            "K_eigenvalues": self.K_eigenvalues,
            "effective_DOF_list": self.effective_DOF_list,
            "Bias_list": self.Bias_list,
            "Variance_list": self.Variance_list,
            "R_theoretical_list": self.R_theoretical_list,
            "Train_theoretical_list": self.Train_theoretical_list
        }
        # write to temp then replace to avoid corrupting existing file
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    
    def load_results(self, path: str) -> dict:
        """
        Usage:
            instance = TheoreticalTestRisk()
            instance.load_results("results/asdf.pkl")
            instance.plot_theoretical_test()
        """
        with open(path, "rb") as f:
            self.loaded = pickle.load(f)

        self.N_QUBITS = self.loaded["N_QUBITS"]
        self.LAMBDA_LIST = self.loaded["LAMBDA_LIST"]
        self.n_train_list = self.loaded["n_train_list"]
        self.K_eigenvalues = self.loaded["K_eigenvalues"]
        self.effective_DOF_list = self.loaded["effective_DOF_list"]
        self.Bias_list = self.loaded["Bias_list"]
        self.Variance_list = self.loaded["Variance_list"]
        self.R_theoretical_list = self.loaded["R_theoretical_list"]
        self.Train_theoretical_list = self.loaded["Train_theoretical_list"]
    
    def plot_theoretical_test(self, bias_variance=False):
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.LAMBDA_LIST)))
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / self.n_train_list
        else:
            ratio_list = 4 ** self.N_QUBITS / self.n_train_list
        
        if bias_variance:
            for i, lam in enumerate(self.LAMBDA_LIST):
                plt.semilogy(ratio_list, self.Bias_list[i], label=f'λ = {lam:.0e}', color=colors[i])
            plt.xlabel(f'$4^{self.N_QUBITS}/N_{{tr}}$', fontsize=12)
            plt.title(f'Theoretical Bias ({self.N_QUBITS} qubits)', fontsize=14)
            plt.legend(title='Regularization λ')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            for i, lam in enumerate(self.LAMBDA_LIST):
                plt.semilogy(ratio_list, self.Variance_list[i], label=f'λ = {lam:.0e}', color=colors[i])
            plt.xlabel(f'$4^{self.N_QUBITS}/N_{{tr}}$', fontsize=12)
            plt.title(f'Theoretical Variance ({self.N_QUBITS} qubits)', fontsize=14)
            plt.legend(title='Regularization λ')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        for i, lam in enumerate(self.LAMBDA_LIST):
            plt.semilogy(ratio_list, self.R_theoretical_list[i], label=f'λ = {lam:.0e}', color=colors[i])
        plt.xlabel(f'$4^{self.N_QUBITS}/N_{{tr}}$', fontsize=12)
        plt.title(f'Theoretical Test Risk ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend(title='Regularization λ')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_theoretical_test_xaxis_lambda(self):
        # Plot theoretical test risk with x-axis as lambda
        # Set `self.n_train_list` by yourself before calling `theoretical_test_risk_lambda`
        # e.g., self.n_train_list = [0.5 * 4**N_QUBITS, 1.0 * 4**N_QUBITS, 2.0 * 4**N_QUBITS]
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.n_train_list)))
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)

        for i, ratio in enumerate(ratio_list):
            plt.semilogy(self.LAMBDA_LIST, self.R_theoretical_list.T[i], label=f'γ = {ratio:.2f}', color=colors[i])
        plt.xscale('log')
        plt.xlabel('$\lambda$', fontsize=12)
        plt.title(f'Theoretical Test Risk ({self.N_QUBITS} qubits)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_effective_DOF(self):
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(self.LAMBDA_LIST)))
        if self.TPA:
            ratio_list = 3 ** self.N_QUBITS / np.array(self.n_train_list)
        else:
            ratio_list = 4 ** self.N_QUBITS / np.array(self.n_train_list)
        
        for i, lambda_val in enumerate(self.LAMBDA_LIST):
            plt.plot(ratio_list, self.effective_DOF_list[i], label=f'λ = {lambda_val:.0e}', color=colors[i])

        plt.xlabel(f'$\gamma = 4^{self.N_QUBITS}/N_{{tr}}$', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(-0.1, 1.3)
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', title='Regularization λ', fontsize=10)
        plt.title(f'Normalized Effective DOF ({self.N_QUBITS} qubits)', fontsize=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def eigen_spectrum(self):
        plt.semilogy(np.sort(self.K_eigenvalues)[::-1], marker='.', linestyle='none', markersize=1)
        plt.xlabel('Index', fontsize=12)
        plt.title(f'Eigenvalue Spectrum of Σ ({self.N_QUBITS} qubits)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def theoretically_test_risk_optimal_labmda():
        pass
