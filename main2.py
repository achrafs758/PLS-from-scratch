from pls import pls
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

df = pd.DataFrame(load_diabetes().data)
df["Y"] = load_diabetes().target

T = pls(df.iloc[:, 0:df.shape[1]-1], df.iloc[:, df.shape[1]-1], p_value_threshold=0.05, correlation_threshold=0.5)
print(T)

X = df.iloc[:, 0:df.shape[1]-1]
y = df.iloc[:, df.shape[1]-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_before = LinearRegression().fit(X_train, y_train)
print(r2_score(model_before.predict(X_test), y_test))

t = T["Y"]
T = T.iloc[:, 0:T.shape[1]-1]
T_train, T_test, t_train, t_test = train_test_split(T, t, test_size=0.2, random_state=42)
model_after = LinearRegression().fit(T_train, t_train)
print(r2_score(model_after.predict(T_test), t_test))