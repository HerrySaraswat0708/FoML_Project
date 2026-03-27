from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
import joblib 

X = np.load('data/X.npy')
y = np.load('data/y.npy')

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr = LinearRegression()

lr.fit(X_train,Y_train)
joblib.dump(lr, "C:\\Users\\LENOVO\\AqSolDB\\models\\lr.pkl")
y_preds = lr.predict(X_test)

score = r2_score(y_preds,Y_test)
mse = mean_squared_error(y_preds,Y_test)
print('R2 score : ',score)
print('MSE :',mse)
