from sklearn.tree import DecisionTreeClassifier

def train_model(X, X_scaled, y):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model
