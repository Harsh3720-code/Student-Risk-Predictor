import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw student dataset & create the target variable.

    :param path (str): Path to the CSV file.

    :return:pd.DataFrame - Dataframe with target column 'at_risk'.

    """

    df = pd.read_csv(path, sep=";")
    # Create target variable
    df['at_risk'] =  (df["G3"] < 10).astype(int)

    return df

if __name__ == "__main__":
    df = load_raw_data("../data/raw/student-mat.csv")
    print(df.head())
    print(df['at_risk'].value_counts())

