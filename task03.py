import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('Housing.csv')

#dataset info
print(df.head())
print(df.info())

df.dropna(inplace=True)

# Simple Linear Regression 
X_simple = df[['area']]
y = df['price']

# Multiple Linear Regression 
X_multiple = df[['area', 'bedrooms', 'bathrooms']]

# Use train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

model_simple = LinearRegression()
model_simple.fit(X_simple, y)

# Plot
plt.scatter(X_simple, y, color='blue')
plt.plot(X_simple, model_simple.predict(X_simple), color='red')
plt.title('House Price vs Area')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()

print("Intercept:", model.intercept_)
print("Coefficients:", list(zip(X_multiple.columns, model.coef_)))
