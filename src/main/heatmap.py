import matplotlib.pyplot as plt
import pandas as pd
import seaborn


def load_data(subject="math"):
    filename = "student-por.csv"
    # mathematics grades
    if subject == "math":
        filename = "student-mat.csv"
    data = pd.read_csv("../../data/" + filename, sep=';')
    return data


if __name__ == "__main__":
    data = load_data()
    print(data)
    corr = data.corr()
    cmap = seaborn.diverging_palette(230, 20, as_cmap=True)
    plt.figure(figsize=(20, 5))
    seaborn.heatmap(corr, annot=True, cmap=cmap)
    plt.show()
