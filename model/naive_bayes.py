from sklearn.naive_bayes import GaussianNB

def train_model(X, X_scaled, y):
    model = GaussianNB()
    model.fit(X_scaled, y)
    return model
