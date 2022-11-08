from plsdsm import PLS #avaiable on pypi
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("Data_Cornell.csv", sep=";")
# df = pd.read_excel("chenille_Thomasson.csv", skiprows=1)
# df = pd.read_excel("poids_taille_data.xlsx")

X = df.iloc[:, 0:df.shape[1]-1]
y = df.iloc[:, df.shape[1]-1]

T = PLS(p_value_threshold=0.05, correlation_threshold=0.5).transform(X, y)

print(T)

model_before = LinearRegression().fit(X, y)
print("R2 Score before PLS", r2_score(model_before.predict(X), y))

model_after = LinearRegression().fit(T, y)
print("R2 Score after PLS", r2_score(model_after.predict(T), y))
