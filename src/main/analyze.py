import csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np

# col numbers used for preprocessing
nominal_attributes = [8, 9, 10, 11]
binary_attributes = [0, 1, 3, 4, 5, 15, 16, 17, 18, 19, 20, 21, 22]
numeric_attributes = [2, 6, 7, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29]


def load_data(subject="math", prior_grades=True):
    # portuguese language grades
    filename = "student-por.csv"
    # mathematics grades
    if subject == "math":
        filename = "student-mat.csv"
    with open("../../data/" + filename) as data_file:
        reader = csv.reader(data_file, delimiter=';')
        X, y = [], []
        # number of columns to include in training set
        n_cols = 32
        if not prior_grades:
            n_cols = 30
        for i, row in enumerate(reader):
            # skip first row
            # TODO use first row as label
            if i == 0:
                continue
            X.append(row[:n_cols])
            y.append(row[-1])
        return X, y


# performs normalization on numeric attributes and 1-of-K encoding on nominal attributes
def preprocess_data(X, y, standardize=False, pass_fail=False):
    # convert binary attributes into 0 or 1
    first_row = X[0].copy()
    X_arr = np.array(X)
    for row in X_arr:
        for i, data in enumerate(row):
            if i in binary_attributes:
                # change value according to equality to the entry in the first row
                row[i] = int(data == first_row[i])

    scaler = StandardScaler()
    if standardize:
        for col in numeric_attributes:
            X_arr[:, col] = scaler.fit_transform(X_arr[:, col].reshape(-1, 1)).reshape(-1)
    y_stand = np.array(y)
    if pass_fail:
        for i in range(len(y_stand)):
            y_stand[i] = int(int(y_stand[i]) >= 10)
    elif standardize:
        y_stand = scaler.fit_transform(y_stand.reshape(-1, 1)).reshape(-1)
    # one-hot encoding
    # remove columns with nominal data
    X_hot = np.append(X_arr[:, :8], X_arr[:, 12:], axis=1)
    enc = OneHotEncoder(sparse=False)
    # append X_hot with the encoded attributes
    for i in nominal_attributes:
        X_hot = np.append(X_hot, enc.fit_transform(X_arr[:, i].reshape(-1, 1)), axis=1)
    return X_hot, y_stand


def score(X_test, y_test, model, classify=False):
    if classify:
        return model.score(X_test, y_test)
    else:
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred, squared=False)


def decision_tree(X, y, classify=False, forest=False):
    X_new, y_new = preprocess_data(X, y, pass_fail=classify)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=1)
    model_code = 2 * int(forest) + int(classify)
    if not model_code:
        print("Using Decision Tree Regressor")
        model = DecisionTreeRegressor()
    elif model_code == 1:
        print("Using Decision Tree Classifier")
        model = DecisionTreeClassifier()
    elif model_code == 2:
        print("Using Random Forest Regressor")
        model = RandomForestRegressor()
    else:
        print("Using Random Forest Classifier")
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print(score(X_test, y_test, model, classify=classify))


def svm(X, y, classify=False):
    # TODO set gamma value
    X_new, y_new = preprocess_data(X, y, pass_fail=classify, standardize=True)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=1)
    if classify:
        model = SVC()
    else:
        model = SVR()
    model.fit(X_train, y_train)
    print(score(X_test, y_test, model, classify=classify))


def neural_network(X, y, classify=False):
    # TODO determine optimal number of hidden nodes
    X_new, y_new = preprocess_data(X, y, pass_fail=classify, standardize=True)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=1)
    if classify:
        model = MLPClassifier(hidden_layer_sizes=(8,))
    else:
        model = MLPRegressor(hidden_layer_sizes=(8,))
    model.fit(X_train, y_train)
    print(score(X_test, y_test, model, classify=classify))


if __name__ == "__main__":
    X, y = load_data(prior_grades=True)
    # decision_tree(X, y, classify=True, forest=False)
    neural_network(X, y, classify=False)
