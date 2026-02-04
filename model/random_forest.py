from sklearn.ensemble import RandomForestClassifier

def train_model(X, X_scaled, y):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    return model
