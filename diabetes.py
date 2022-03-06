from sklearn.model_selection import train_test_split
import numpy as np
import shap
import time
from sklearn import linear_model


def print_accuracy(f):
    print(
        "Root mean squared test error = {0}".format(
            np.sqrt(np.mean((f(X_test) - y_test) ** 2))
        )
    )
    time.sleep(0.5)  # to let the print get out before any progress bars


X, y = shap.datasets.diabetes()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# print(type(X_train))
# print(type(X_test))
# X_train = X_train.to_numpy()
# X_test = X_test.to_numpy()

X_train_summary = shap.kmeans(X_train, 10)


lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train, y_train)


print_accuracy(lin_regr.predict)

ex = shap.KernelExplainer(lin_regr.predict, X_train_summary)
shap_values = ex.shap_values(X_test.iloc[0, :])
shap.force_plot(ex.expected_value, shap_values, X_test.iloc[0, :], matplotlib=True)


# shap_values = ex.shap_values(X_test)
# shap.summary_plot(shap_values, X_test)
