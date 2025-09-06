import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split

# type: ignore
x, y = make_regression(n_samples=100, n_features=1, noise=1, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

alpha = 1
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(x_train, y_train)

y_pred_lr = lr_model.predict(x_test)
y_pred_lasso = lasso_model.predict(x_test)

plt.scatter(x_test, y_test, color="black", label="Original Test Data")

plt.plot(
    x_test,
    y_pred_lr,
    color="blue",
    linewidth=2,
    label=f"Linear Regression (Coeff: {lr_model.coef_[0]:.2f})",
)

plt.plot(
    x_test,
    y_pred_lasso,
    color="red",
    linewidth=2,
    label=f"Lasso (alpha={alpha}, Coeff: {lasso_model.coef_[0]:.2f})",
)

plt.title("Comparison of Linear Regression and Lasso")
plt.xlabel("Feature Value")
plt.ylabel("Target Value")
plt.legend()
plt.grid()
plt.show()
