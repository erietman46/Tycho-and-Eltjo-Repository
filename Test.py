from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# -----------------------
# 1. Create Synthetic Data
# -----------------------
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 2. Build a Pipeline
# -----------------------
pipeline = Pipeline([
    ("pca", PCA()),  # reduce dimensionality
    ("rf", RandomForestClassifier(random_state=42))
])

# -----------------------
# 3. Hyperparameter Search
# -----------------------
param_grid = {
    "pca__n_components": [5, 10, 15],
    "rf__n_estimators": [50, 100],
    "rf__max_depth": [None, 5, 10]
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# -----------------------
# 4. Evaluate
# -----------------------
print("Best Parameters:", grid.best_params_)
y_pred = grid.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
