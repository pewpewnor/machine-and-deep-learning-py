import pandas as pd

iris_dataset = "./iris_dataset/iris.data"


def main():
    df = pd.read_csv(iris_dataset, header=None)
    print(df.iloc[0:100, 4].values)


if __name__ == "__main__":
    main()
