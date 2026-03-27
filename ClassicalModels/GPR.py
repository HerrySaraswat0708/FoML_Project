import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib 

X = np.load("data/X.npy")   
y = np.load("data/y.npy")   
y = y.reshape(-1)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

gpr = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,             
    normalize_y=True,
    n_restarts_optimizer=5, 
    random_state=42
)

gpr.fit(X_train, y_train)
joblib.dump(gpr, "C:\\Users\\LENOVO\\AqSolDB\\models\\gpr.pkl")
print("Optimized Kernel:")
print(gpr.kernel_)


y_pred, y_std = gpr.predict(X_test, return_std=True)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPerformance:")
print("MSE:", mse)
print("R2 Score:", r2)


if X.shape[1] == 1:
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    y_plot, y_std_plot = gpr.predict(X_plot, return_std=True)

    plt.figure(figsize=(10, 6))
    
    # Training data
    plt.scatter(X_train, y_train, c='blue', label='Train Data')
    
    # Prediction mean
    plt.plot(X_plot, y_plot, 'r-', label='Prediction')
    
    # Confidence interval
    plt.fill_between(
        X_plot.ravel(),
        y_plot - 2*y_std_plot,
        y_plot + 2*y_std_plot,
        alpha=0.2,
        label='Confidence Interval (±2σ)'
    )
    
    plt.legend()
    plt.title("Gaussian Process Regression")
    plt.show()