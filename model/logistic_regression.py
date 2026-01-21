from sklearn.linear_model import LogisticRegression

def train_model(X, X_scaled, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model
