import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load UCI Wine dataset through scikit-learn
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
print("WINE CLASSIFICATION: NAIVE BAYES, KNN AND DECISION TREE")
print("Samples:", X.shape[0], "Features:", X.shape[1],
      "Classes:", list(wine.target_names))
print("Class counts:", np.bincount(y).tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.30, stratify=y, random_state=42)
models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Decision Tree (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42)
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv = cross_val_score(model, X, y, cv=folds, scoring="accuracy")
    results.append([name, train_acc, test_acc, cv.mean(), cv.std()])

table = pd.DataFrame(results, columns=["Model", "Train Acc", "Test Acc", "CV Mean", "CV Std"])
print("\nMODEL COMPARISON\n", table.round(4).to_string(index=False))
best_name = table.sort_values(["CV Mean", "Test Acc"], ascending=False).iloc[0]["Model"]
best = models[best_name]
pred = best.predict(X_test)
print("\nSelected model:", best_name)
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
print("Classification report:\n",
      classification_report(y_test, pred, target_names=wine.target_names, zero_division=0))

print("\nEFFECT OF k ON KNN (5-FOLD CV)")
for k in [1, 3, 5, 7, 11, 15]:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    print("k =", k, "accuracy =", round(cross_val_score(knn, X, y, cv=folds).mean(), 4))

tree = models["Decision Tree (depth=3)"]
tree.fit(X_train, y_train)
print("\nDECISION TREE RULES\n", export_text(tree, feature_names=list(X.columns)))
