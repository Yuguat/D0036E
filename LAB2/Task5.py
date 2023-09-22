#overfitting:
#under
# if data is poor fit to training and test data, underfitting
#if data is good fit for training but poor fit with test set over fittin.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
def reading_csv(csv_file):
    f = open(csv_file, "r")
    age=[]
    income=[]
    region=[]
    for line in f.readlines()[1:]:
        V1=line.replace("\n","")
        V2=V1.split(",")
        region.append(V2[1])
        age.append(V2[2])
        income.append(float(V2[3]))

    age_c=[]
    for item in age:
        if "+" in item.split()[0]:
            item=item.replace("+","")
        temp=int(item.split()[0])
        age_c.append(temp)

    region_num=[]
    region_name=[]
    for item in region:
        region_num.append(int(item.split()[0]))
        region_name.append(item.split()[1])

    a=list(range(0,(len(age))))
    d = {"region_num":region_num,'region': region_name,'age': age_c,'income': income}
    df=pd.DataFrame(data=d, index=a)

    #print(df)
    return df

def linear_regression(Data_F):
    X=Data_F["age"].to_numpy().T
    Y=Data_F["income"].to_numpy().T
    X_b=np.c_[np.ones((len(X),1)),X]
    t1 = X_b.T.dot(X_b)
    t2 = np.linalg.inv(t1)  # What does inv do?
    t3 = t2.dot(X_b.T)
    theta_best = t3.dot(Y)
    return theta_best
def Seperate_Data(Data_F):
    D=Data_F
    Validation_Data=D.sample(frac=0.2, random_state = 42)
    Training_Data=D.drop(Validation_Data.index)
    return Validation_Data, Training_Data

def Make_prediction(theta,X_new_l):
    X_new = np.array(X_new_l)
    len(X_new)
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
    # We dot product with theta to get the prediction
    y_predict = X_new_b.dot(theta)
    return X_new,y_predict
def MSE_Calculation(theta,Data):
    X_i=Data["age"].to_numpy().T
    Y_i=Data["income"].to_numpy().T
    X_new, Y_predict = Make_prediction(theta,X_i)
    MSE=sum((Y_i-Y_predict)**2)/len(X_i)
    return MSE



csv_file="inc_utf.csv"
df=reading_csv(csv_file)

Data_Group= df.groupby(["age"], as_index=False)["income"].mean()
print(Data_Group)
Val_Data,T_Data=Seperate_Data(Data_Group)

#Degree 2
poly_features=PolynomialFeatures(degree=2,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 2 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=np.linspace(X_min, X_max, len(X)).reshape(len(X), 1)
print(X_new)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Degree2 : MSE  "+str(MSE)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

#Degree 3
poly_features=PolynomialFeatures(degree=3,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 3 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=np.linspace(X_min, X_max, len(X)).reshape(len(X), 1)
print(X_new)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Degree3 : MSE  "+str(MSE)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

#Degree 4
poly_features=PolynomialFeatures(degree=4,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 4 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=np.linspace(X_min, X_max, len(X)).reshape(len(X), 1)
print(X_new)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Degree4 : MSE  "+str(MSE)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

#Degree 5
poly_features=PolynomialFeatures(degree=5,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 5 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=np.linspace(X_min, X_max, len(X)).reshape(len(X), 1)
print(X_new)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Degree5 : MSE  "+str(MSE)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

#Degree 9
poly_features=PolynomialFeatures(degree=9,include_bias=False)

X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 9 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=np.linspace(X_min, X_max, len(X)).reshape(len(X), 1)
print(X_new)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Degree9 : MSE  "+str(MSE)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()




#Degree 10
poly_features=PolynomialFeatures(degree=10,include_bias=False)

X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 10 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=np.linspace(X_min, X_max, len(X)).reshape(len(X), 1)
print(X_new)
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Degree10 : MSE  "+str(MSE)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

