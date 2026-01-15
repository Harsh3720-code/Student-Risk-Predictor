import numpy as np
import pandas as pd

from src.data_loader import load_raw_data

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer that:
    - Scales numerical features
    - One-hot encodes categorical features

    :param: X (pd.DataFrame) - Feature DataFrame

    :returns: ColumnTransformer: Preprocessing pipeline
    """

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Numerical Pipeline
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    # Categorical Pipeline
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])

    # Combine both
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numerical_cols), ("cat", categorical_transformer, categorical_cols)])

    return preprocessor


def main():
    print("Loading raw data...")

    df = load_raw_data("../data/raw/student-mat.csv")

    print("Initial shape: ", df.shape)
    print("Target distribution: ")
    print(df['at_risk'].value_counts())

    # Drop leakage columns
    print("Dropping Leakage columns: G1, G2, G3")
    df_model = df.drop(columns=['G1', 'G2', 'G3'])

    # Separate X & Y columns
    X = df_model.drop(columns=['at_risk'])
    y = df_model['at_risk']

    print("X Shape: ", X.shape)
    print("y Shape: ", y.shape)

    # Build preprocessor
    print("Building preprocessor pipeline...")
    preprocessor = build_preprocessor(X)

    # Train-Test Split
    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    # Apply preprocessing
    print("Fitting & Transforming data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Processed shapes:")
    print("X_train_processed: ", X_train_processed.shape)
    print("X_test_processed: ", X_test_processed.shape)

    # Ensure y are clean 1D arrays
    y_train = np.array(y_train).reshape(-1).astype(int)
    y_test = np.array(y_test).reshape(-1).astype(int)

    print("y_train shape: ", y_train.shape, "unique:", np.unique(y_train))
    print("y_test shape: ", y_test.shape, "unique:", np.unique(y_test))

    # Save outputs
    print("Saving processed array...")

    np.save("../data/preprocessed/X_train_processed.npy", X_train_processed)
    np.save("../data/preprocessed/y_train.npy", y_train)
    np.save("../data/preprocessed/X_test_processed.npy", X_test_processed)
    np.save("../data/preprocessed/y_test.npy", y_test)

    print("Preprocessing completed successfully.!")

if __name__ == "__main__":
    main()















