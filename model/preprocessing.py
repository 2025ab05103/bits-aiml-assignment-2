import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_file):
    df = pd.read_csv(csv_file)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )

    target_col = "absenteeism_time_in_hours"
    if target_col not in df.columns:
        raise ValueError("Target column not found")

    df["absenteeism_class"] = (df[target_col] >= 8).astype(int)
    df.drop(target_col, axis=1, inplace=True)

    X = df.drop("absenteeism_class", axis=1)
    y = df["absenteeism_class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, y
