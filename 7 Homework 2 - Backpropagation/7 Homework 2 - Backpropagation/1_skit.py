import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from data_helper import  *

RANDOM_STATE = 17
np.random.seed(RANDOM_STATE)


def eval_model(model, X, t, keymsg, squared = False):
    t_pred = model.predict(X)
    error = mean_squared_error(t, t_pred, squared=squared)  # rmse

    print(f'\t Error of {keymsg}: {error}')
    return error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Regressors Homework')

    parser.add_argument('--dataset', type=str, default='data2_200x30.csv')
    parser.add_argument('--preprocessing', type=int, default=1,
                        help='0 for no processing, 1 for min/max scaling and 2 for standrizing')

    args = parser.parse_args()

    X, t, X_train, t_train, X_val, t_val = load_data(args.dataset)

    preprocess_option = args.preprocessing

    X_train_p, X_val_p = preprocess_data(X_train, X_val, preprocess_option=preprocess_option)

    hidden_layer_sizes = (5, 5, 5)
    model = MLPRegressor(hidden_layer_sizes, random_state=RANDOM_STATE, max_iter=10000)
    model = model.fit(X_train_p, t_train.reshape(-1))

    eval_model(model, X_val_p, t_val, 'NN')

    # Error of NN: 12.635787980691891


