# ============================================================
# Problem #1: Sklearn - Neural Network Regression
# Homework 2 - Backpropagation
# ============================================================
# GOAL: Find the best Neural Network architecture
#       that gives the lowest validation RMSE
# ============================================================

import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from data_helper import *

# Fix randomness so results are reproducible every run
RANDOM_STATE = 17
np.random.seed(RANDOM_STATE)


def eval_model(model, X, t, keymsg):
    """
    Evaluates a trained model on given data.
    
    Args:
        model   : trained MLPRegressor model
        X       : input features
        t       : true target values
        keymsg  : label for printing
    
    Returns:
        rmse : Root Mean Squared Error (lower = better)
    """
    t_pred = model.predict(X)           # get predictions
    mse    = mean_squared_error(t, t_pred)  # mean squared error
    rmse   = np.sqrt(mse)               # root mean squared error
    print(f'\t Error of {keymsg}: {rmse}')
    return rmse


if __name__ == '__main__':

    # ── 1. Read arguments from terminal ──────────────────────
    parser = argparse.ArgumentParser(description='Regressors Homework')
    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    parser.add_argument('--preprocessing', type=int, default=1,
                        help='0=no scaling, 1=MinMax, 2=Standardize')
    args = parser.parse_args()

    # ── 2. Load data ──────────────────────────────────────────
    # Reads CSV → fills missing values with median
    # Splits: first 100 rows = train, last 100 rows = validation
    X, t, X_train, t_train, X_val, t_val = load_data(args.dataset)

    # ── 3. Preprocess data ────────────────────────────────────
    # Fit scaler on train only → apply to val (no data leakage)
    preprocess_option = args.preprocessing
    X_train_p, X_val_p = preprocess_data(X_train, X_val,
                                          preprocess_option=preprocess_option)

    # ── 4. Baseline Model ─────────────────────────────────────
    # Architecture: Input(30) → [5] → [5] → [5] → Output(1)
    print("\n📊 Baseline Model:")
    print("-" * 50)
    baseline_model = MLPRegressor(
        hidden_layer_sizes=(5, 5, 5),   # 3 hidden layers, 5 neurons each
        random_state=RANDOM_STATE,
        max_iter=10000
    )
    baseline_model.fit(X_train_p, t_train.reshape(-1))
    baseline_rmse = eval_model(baseline_model, X_val_p, t_val,
                               'Baseline (5,5,5)')

    # ── 5. Best Model ─────────────────────────────────────────
    # Found by experimenting with:
    # - Different architectures: (5,5,5), (100,), (50,25), (64,32,16)...
    # - Different activations  : relu, tanh
    # - Different solvers      : adam, sgd
    # Winner: (64,32,16) + tanh + adam
    print("\n🏆 Best Model:")
    print("-" * 50)
    best_model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),  # pyramid: wide → narrow
        activation='tanh',                # better than relu for regression
        solver='adam',                    # adaptive optimizer
        random_state=RANDOM_STATE,
        max_iter=50000,                   # more iterations for bigger network
        tol=1e-5                          # stricter convergence
    )
    best_model.fit(X_train_p, t_train.reshape(-1))
    best_rmse = eval_model(best_model, X_val_p, t_val,
                           'Best (64,32,16)+tanh+adam')

    # ── 6. Summary ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)
    print(f"  Baseline RMSE : {baseline_rmse:.4f}")
    print(f"  Best RMSE     : {best_rmse:.4f}")
    print(f"  Improvement   : {baseline_rmse - best_rmse:.4f} ✅")
    print("=" * 50)