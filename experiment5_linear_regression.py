import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate house-price data (price in lakhs)
np.random.seed(10)
n = 150
area = np.random.randint(500, 3000, n)
bedrooms = np.random.randint(1, 5, n)
age = np.random.randint(0, 30, n)
distance = np.round(np.random.uniform(1, 25, n), 1)
price = 15 + .045*area + 4*bedrooms - .6*age - 1.2*distance + np.random.normal(0, 6, n)
houses = pd.DataFrame({"area_sqft":area, "bedrooms":bedrooms, "age_years":age,
                       "distance_km":distance, "price_lakhs":price.round(2)})

print("HOUSE PRICE PREDICTION")
print(houses.head().to_string(index=False))
print("\nCorrelation with price:")
print(houses.corr(numeric_only=True)["price_lakhs"].drop("price_lakhs").round(3))

X, y = houses.drop(columns="price_lakhs"), houses["price_lakhs"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

# Simple regression using area only
simple = LinearRegression().fit(X_train[["area_sqft"]], y_train)
simple_pred = simple.predict(X_test[["area_sqft"]])
print("\nSIMPLE LINEAR REGRESSION")
print("Intercept:", round(simple.intercept_, 4), "Slope:", round(simple.coef_[0], 4))
print("Test R2:", round(r2_score(y_test, simple_pred), 4))

# Multiple regression using all features
model = LinearRegression().fit(X_train, y_train)
print("\nMULTIPLE REGRESSION COEFFICIENTS")
print("Intercept:", round(model.intercept_, 4))
for name, coef in zip(X.columns, model.coef_):
    print(name, round(coef, 4))

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print("\nTEST METRICS")
print("MAE:", round(mean_absolute_error(y_test, pred), 4))
print("MSE:", round(mse, 4), "RMSE:", round(np.sqrt(mse), 4))
print("R2:", round(r2_score(y_test, pred), 4))

new_house = pd.DataFrame([[1800, 3, 5, 8.0]], columns=X.columns)
print("\nPredicted price for new house:",
      round(model.predict(new_house)[0], 2), "lakhs")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
xs = np.linspace(500, 3000, 100).reshape(-1, 1)
ax[0].scatter(X_train["area_sqft"], y_train, s=15, label="Training data")
ax[0].plot(xs, simple.predict(pd.DataFrame(xs, columns=["area_sqft"])),
           color="red", label="Fitted line")
ax[0].set(xlabel="Area (sqft)", ylabel="Price (lakhs)", title="Simple Linear Regression")
ax[0].legend()
ax[1].scatter(pred, y_test-pred, color="purple")
ax[1].axhline(0, color="black", linestyle="--")
ax[1].set(xlabel="Predicted price", ylabel="Residual", title="Residual Plot")
plt.tight_layout()
plt.savefig("exp05_regression.png", dpi=150)
plt.show()
