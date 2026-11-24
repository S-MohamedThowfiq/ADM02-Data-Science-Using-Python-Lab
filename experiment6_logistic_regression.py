import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score, roc_curve)

# Generate telecom customer churn data
np.random.seed(3)
n = 400
tenure = np.random.randint(1, 72, n)
monthly = np.round(np.random.uniform(200, 1500, n), 0)
complaints = np.random.poisson(1.2, n)
contract = np.random.choice([0, 1], n, p=[.55, .45])  # 1 = yearly contract
logit = -1 - .06*tenure + .002*monthly + .7*complaints - 1.3*contract
churn = (np.random.rand(n) < 1/(1+np.exp(-logit))).astype(int)
data = pd.DataFrame({"tenure_months":tenure, "monthly_bill":monthly,
                     "complaints":complaints, "yearly_contract":contract, "churn":churn})
print("CUSTOMER CHURN PREDICTION")
print(data.head().to_string(index=False))
print("Churn distribution:", data.churn.value_counts().to_dict())

X, y = data.drop(columns="churn"), data.churn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.25, stratify=y, random_state=0)

# Scaling is learned from training data within the pipeline
model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
lr = model.named_steps["logisticregression"]
print("\nCOEFFICIENTS AND ODDS RATIOS")
for name, coef in zip(X.columns, lr.coef_[0]):
    print(name, "coefficient =", round(coef, 4), "odds ratio =", round(np.exp(coef), 3))
print("Intercept:", round(lr.intercept_[0], 4))

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print("\nTEST METRICS")
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, pred, zero_division=0), 4))
print("F1:", round(f1_score(y_test, pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

cv = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()),
                     X, y, cv=5, scoring="accuracy")
print("\n5-fold CV:", np.round(cv, 4), "Mean:", round(cv.mean(), 4),
      "Std:", round(cv.std(), 4))

new_customer = pd.DataFrame([[4, 1250, 3, 0]], columns=X.columns)
print("New customer's churn probability:",
      round(model.predict_proba(new_customer)[0, 1], 3))

fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure(figsize=(5.5, 4.5))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, prob):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Churn Model"); plt.legend(); plt.tight_layout()
plt.savefig("exp06_roc.png", dpi=150)
plt.show()
