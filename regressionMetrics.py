from sklearn.metrics import mean_absolute_error

# Example true values and predictions
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
print("Mean Absolute Error (MAE):", mae)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error (MSE):", mse)
import numpy as np

rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print("R-squared (R2):", r2)
