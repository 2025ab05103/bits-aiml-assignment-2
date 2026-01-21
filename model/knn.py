from sklearn.neighbors import KNeighborsClassifier

def train_model(X, X_scaled, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, y)
    return model
