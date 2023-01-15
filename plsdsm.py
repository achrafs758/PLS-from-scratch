import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
import statsmodels.api as sm


class PLS:
    def __init__(self, p_value_threshold=0.05, correlation_threshold=0.5):
        self.p_value_threshold = p_value_threshold
        self.correlation_threshold = correlation_threshold

    def transform(self, X, y):
        df = pd.DataFrame(X)
        df['Y'] = y

        corr = df.corr()
        
        df_scaled = preprocessing.scale(df)
        df_scaled = pd.DataFrame(df_scaled)
        
        n_features = X.shape[1] - 1
        corr_y = corr.iloc[n_features][0:n_features]
        corr_yabs = abs(corr_y)

        Xi = []
        for i, elem in enumerate(corr_yabs):
            if elem > self.correlation_threshold:
                Xi.append(i)

        sum1 = 0
        sum2 = 0
        for i in Xi:
            sum1 += corr["Y"][:-1][i] * df_scaled[i]
            sum2 += corr["Y"][:-1][i]**2

        T1 = (1 / math.sqrt(sum2)) * sum1
        T = pd.DataFrame({"T1": T1})

        Y_scaled = df_scaled[n_features]

        j = 2
        while True:
            model = []
            for i in range(n_features):
                X_ = pd.concat([df_scaled[i], T], axis=1)
                model.append(sm.OLS(Y_scaled, X_).fit())

            col2 = []
            for i in range(len(df_scaled.columns)-1):
                p_value = model[i].pvalues.tolist()[0]
                if p_value < self.p_value_threshold:
                    col2.append(i)
            
            if len(col2) == 0:
                break
            
            residu = []
            for i in col2:
                dt = pd.DataFrame(T)
                model = sm.OLS(df_scaled[i], dt).fit()
                residu.append(model.resid)
            X1n = []
            for r in residu:
                X1n.append(r/np.var(r))

            model2 = []
            for i in range(len(X1n)):
                dt = T.copy()
                dt[1] = X1n[i]
                model2.append(sm.OLS(Y_scaled, dt).fit())
            
            coefs = np.array([model2[i].params.tolist()[1] for i in range(len(X1n))])
            
            sum1 = 0
            sum2 = 0
            for i in range(len(residu)):
                sum1 += coefs[i] * residu[i]
                sum2 += coefs[i]**2

            T["T"+str(j)] = ((1 / math.sqrt(sum2)) * sum1)

            j += 1
        
        T["Y"] = y
        
        return T
