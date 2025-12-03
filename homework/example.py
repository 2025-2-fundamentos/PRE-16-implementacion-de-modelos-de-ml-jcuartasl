import pandas as pd

from homework import linear_regression



def main():
    data = pd.read_csv('./files/input/data.csv')
    X = data[['x1', 'x2']].values
    y = data['y'].values

    model = linear_regression.LinearRegression(num_epochs=5000, learning_rate=0.01)
    model.fit(X, y)

    predictions = model.predict(X)

    for y_true, y_pred in zip(y, predictions):
        print(f'{y_true:8.4f}, {y_pred:8.4f}')

if __name__ == '__main__':
    main()