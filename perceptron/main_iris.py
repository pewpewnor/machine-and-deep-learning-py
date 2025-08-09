import pandas as pd

iris_dataset = "./iris_dataset/iris.data"


def main():
    df = pd.read_csv(iris_dataset, header=None)
    labels = df.iloc[:, 4].values
    features = df.iloc[:, :3].values
    print(labels, features)


if __name__ == "__main__":
    main()
