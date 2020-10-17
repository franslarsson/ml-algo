
import numpy as np

from linear_regression import LinearRegression


def test_linear_regression():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([10, 20, 30, 40, 50]).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    assert np.isclose(model.beta, np.array([0, 10]).reshape(-1, 1)).all()

    X_test = np.array([-1, -2]).reshape(-1, 1)
    y_pred = model.predict(X_test)
    assert np.isclose(y_pred, np.array([-10, -20]).reshape(-1, 1)).all()

    X = np.array([-2, -1, 1, 2]).reshape(-1, 1)
    y = np.array([9, 7, 3, 1]).reshape(-1, 1)
    model.fit(X, y)
    assert np.isclose(model.beta, np.array([5, -2]).reshape(-1, 1)).all()
