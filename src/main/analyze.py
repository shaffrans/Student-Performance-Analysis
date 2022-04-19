import csv
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        X, y, labels = [], [], []
        # number of columns to include in training set
        n_cols = 32
        if not prior_grades:
            n_cols = 30
        for i, row in enumerate(reader):
            # use first row as label
            if i == 0:
                labels = row[:n_cols]
                continue
            X.append([(int(n) if j in numeric_attributes else n) for j, n in enumerate(row[:n_cols])])
            y.append(int(row[-1]))
        return X, y, labels


# performs normalization on numeric attributes and 1-of-K encoding on nominal attributes
def preprocess_data(X, y, labels=None, standardize=False, pass_fail=False):
    # convert binary attributes into 0 or 1
    labels_new = labels.copy() if labels else []
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
            y_stand[i] = int(y_stand[i] >= 10)
    elif standardize:
        y_stand = scaler.fit_transform(y_stand.reshape(-1, 1)).reshape(-1)
    # one-hot encoding
    # remove columns with nominal data
    X_hot = np.append(X_arr[:, :8], X_arr[:, 12:], axis=1)
    enc = OneHotEncoder(sparse=False)
    # append X_hot with the encoded attributes
    for i in nominal_attributes:
        X_hot = np.append(X_hot, enc.fit_transform(X_arr[:, i].reshape(-1, 1)), axis=1)
        # update labels
        if labels:
            for name in enc.get_feature_names_out():
                labels_new.append(labels_new[i] + ':' + name[3:])
    labels_out = []
    if labels:
        for i, label in enumerate(labels_new):
            if i not in nominal_attributes:
                labels_out.append(label)
    return X_hot, y_stand, labels_out


def score(X_test, y_test, model, classify=False):
    if classify:
        return model.score(X_test, y_test)
    else:
        y_pred = model.predict(X_test)
        return mean_squared_error(y_test, y_pred, squared=False)


# some of this code was copied from scikit-learn.org
def feature_importances(X_test, y_test, model, labels):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean, index=labels)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def decision_tree(X, y, labels, classify=False, forest=False):
    X_new, y_new, labels_new = preprocess_data(X, y, labels, pass_fail=classify)
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
    feature_importances(X_test, y_test, model, labels_new)


def svm(X, y, classify=False):
    # TODO set gamma value
    X_new, y_new, _ = preprocess_data(X, y, pass_fail=classify, standardize=True)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=1)
    if classify:
        model = SVC()
    else:
        model = SVR()
    model.fit(X_train, y_train)
    print(score(X_test, y_test, model, classify=classify))


def neural_network(X, y, classify=False):
    # TODO determine optimal number of hidden nodes
    X_new, y_new, _ = preprocess_data(X, y, pass_fail=classify, standardize=True)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=1)
    if classify:
        model = MLPClassifier(hidden_layer_sizes=(8,))
    else:
        model = MLPRegressor(hidden_layer_sizes=(8,))
    model.fit(X_train, y_train)
    print(score(X_test, y_test, model, classify=classify))


def naive_predictor(X, y, classify=False):
    X_new, y_new, _ = preprocess_data(X, y, pass_fail=classify)
    X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.33, random_state=1)
    # if grade information is available, use that to predict performance
    grades_available = False
    if len(X[0]) > 30:
        grades_available = True
    prediction = 0
    if not grades_available:
        if classify:
            prediction = np.round(np.average(y_train))
        else:
            prediction = np.average(y_train)
    # set the prediction according to the previous term's grade
    if grades_available:
        if classify:
            y_pred = [int(float(grade) >= 10) for grade in X_test[:, 27]]
            print("grade available, classify")
        else:
            y_pred = X_test[:, 27].reshape(-1)
            print("grade available, regress")
    # if grade information is not available, predict the most likely outcome for everyone
    else:
        y_pred = [prediction] * len(y_test)
        print("grade not available")
    if classify:
        print(sum([int(i == j) for i, j in zip(y_pred, y_test)]) / len(y_test))
    else:
        print(mean_squared_error(y_test, y_pred, squared=False))


def write_processed_data(subject="math"):
    X, y, labels = load_data(subject=subject, prior_grades=True)
    X, y, labels = preprocess_data(X, y, labels, False, False)
    data = np.append(X[:, :28], np.append(y.reshape(-1, 1), X[:, 28:], axis=1), axis=1)
    labels = np.array(labels[:28] + ["G3"] + labels[28:]).reshape(1, -1)
    data = np.append(labels, data, axis=0)
    writer = csv.writer(open("../../data/preprocessed_" + subject + ".csv", "w+"))
    writer.writerows(data)


if __name__ == "__main__":
    X, y, labels = load_data(prior_grades=False, subject="math")
    decision_tree(X, y, labels, classify=False, forest=False)
    # naive_predictor(X, y, classify=True)
    # write_processed_data(subject="portuguese")
