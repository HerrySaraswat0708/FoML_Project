from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score

X = np.load('data/X.npy')
y = np.load('data/y.npy')

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)

lr = LinearRegression(lr=1e-4)

lr.fit(X_train,Y_train)

y_preds = lr.predict(X_test)

score = r2_score(y_preds,y)
print('R2 score : ',score)
