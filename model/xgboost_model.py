from xgboost import XGBClassifier

def train_model(X, X_scaled, y):
    model = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    model.fit(X, y)
    return model
