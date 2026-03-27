import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
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

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), 
    activation='relu',
    solver='adam',
    alpha=1e-4,            
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=1e-3,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
    verbose=True
)


mlp.fit(X_train, y_train)
joblib.dump(mlp, "C:\\Users\\LENOVO\\AqSolDB\\models\\mlp.pkl")

y_pred = mlp.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nPerformance:")
print("MSE:", mse)
print("R2 Score:", r2)


plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()
plt.show()

if X.shape[1] == 1:
    X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)

    y_plot = mlp.predict(X_plot_scaled)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_train[:, 0], y_train, label="Train Data", alpha=0.6)
    plt.plot(X_plot, y_plot, color='red', label="MLP Prediction")
    plt.legend()
    plt.title("MLP Regression Fit")
    plt.show()