# Double Descent in Quantum Kernel Ridge Regression

This repository contains the official simulation code for the paper:
**[Double Descent in Quantum Kernel Ridge Regression](https://arxiv.org/abs/2604.17202)** by Kensuke Kamisoyama, Lento Nagano, and Koji Terashi.

This code explores the generalization dynamics—specifically the double descent phenomenon—of overparameterized Quantum Machine Learning (QML) models. It compares the **empirical test risk** computed via statevector simulations against the **asymptotic theoretical test risk** derived using tools from Random Matrix Theory (RMT).

## 📁 Repository Structure

```text
.
├── QKRR.py                           # Core library (Data, Kernels, Numerics, and Theory)
├── QKRR_mnist_3qubits_HEA.ipynb      # HEA on Fashion-MNIST (3 qubits)
├── QKRR_mnist_3qubits_TPA.ipynb      # TPA on Fashion-MNIST (3 qubits)
├── QKRR_synthetic_3qubits.HEA.ipynb  # HEA on Synthetic Data (3 qubits)
├── QKRR_synthetic_3qubits_TPA.ipynb  # TPA on Synthetic Data (3 qubits)
├── QKRR_synthetic_5qubits.HEA.ipynb  # HEA on Synthetic Data (5 qubits)
├── QKRR_synthetic_5qubits_TPA.ipynb  # TPA on Synthetic Data (5 qubits)
├── pyproject.toml / uv.lock          # Python dependencies and environment setup
└── results/                          # Saved outputs (.pkl files) and generated plots (.pdf)
```

## ⚙️ Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management, as indicated by the `uv.lock` and `pyproject.toml` files.

**Using `uv` (Recommended):**
```bash
# Clone the repository
git clone https://github.com/kkensuke/DD_in_QKRR_code.git
cd DD_in_QKRR_code

# Sync the environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

*(Alternatively, you can install the dependencies via standard pip if you prefer: `pip install pennylane scikit-learn numpy scipy matplotlib jupyter`)*

## Usage & Reproducing the Results

To reproduce the figures from the paper, you can run the provided Jupyter notebooks. Each notebook corresponds to a specific combination of dataset, number of qubits, and quantum circuit ansatz.

### Ansatz Configurations in the Code:
* **Hardware-Efficient Ansatz (HEA):** Used by default (`TPA = False`).
* **Tensor Product Ansatz (TPA):** Activated by setting `TPA = True`.

**Note on Execution:** Computing the empirical kernel matrices over multiple iterations and estimating the theoretical fixed-point equations can be computationally expensive. The scripts are designed to cache their progress as `.pkl` files in the `results/` directory. If a `.pkl` file is present, you can load the pre-computed results directly to plot the figures.

## Core Module Overview (`QKRR.py`)

The `QKRR.py` file contains the heavy lifting for the simulations:

* **`load_fashion_mnist` & `SyntheticDataset`:** Functions/Classes to prepare the Fashion-MNIST and synthetic datasets, respectively.
* **`QuantumKernel`:** Implements the quantum feature maps and computes the kernel matrices $K_{ij} = |\langle 0| U^\dagger(x_i) U(x_j) |0\rangle|^2$ using `PennyLane`. 
* **`NumericalTestRisk`:** Sweeps through different model complexity ratios $\gamma = p/N_{tr}$ and regularization parameters $\lambda$ to numerically compute the empirical Mean Squared Error (MSE) using the closed-form dual estimator.
* **`TheoreticalTestRisk`:** Uses the dataset to estimate the population covariance $\Sigma$ and the projected target vector $\beta_*$. It then solves the implicit self-consistent equations (fixed-point equations) from Random Matrix Theory to compute the deterministic equivalent of the test risk without having to train the model. 
