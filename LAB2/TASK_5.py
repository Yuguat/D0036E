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
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_T=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Training Degree2 : MSE  "+str(MSE_T)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()
#validation Dataset
X=Val_Data["age"].to_numpy().T
Y=Val_Data["income"].to_numpy().T
X_b=np.c_[X]
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_val=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Validation Degree2 : MSE  "+str(MSE_val)), rotation=0, fontsize=18)
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
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_T=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Training Degree3 : MSE  "+str(MSE_T)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()
#validation Dataset
X=Val_Data["age"].to_numpy().T
Y=Val_Data["income"].to_numpy().T
X_b=np.c_[X]
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_val=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Validation Degree3 : MSE  "+str(MSE_val)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

#Degree 6
poly_features=PolynomialFeatures(degree=6,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 6 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_T=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Training Degree6 : MSE  "+str(MSE_T)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()
#validation Dataset
X=Val_Data["age"].to_numpy().T
Y=Val_Data["income"].to_numpy().T
X_b=np.c_[X]
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_val=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Validation Degree6 : MSE  "+str(MSE_val)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()

#Degree 8
poly_features=PolynomialFeatures(degree=8,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 8 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_T=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Training Degree8 : MSE  "+str(MSE_T)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()
#validation Dataset
X=Val_Data["age"].to_numpy().T
Y=Val_Data["income"].to_numpy().T
X_b=np.c_[X]
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_val=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Validation Degree8 : MSE  "+str(MSE_val)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()


def plot_learning_curves(model, X, y):
     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
     train_errors, val_errors = [], []
     for m in range(1, len(X_train)):
            model.fit(X_train[:m], y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
            val_errors.append(mean_squared_error(y_val, y_val_predict))
     plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
     plt.plot(np.sqrt(val_errors), "b-", linewidth=2, label="val")
     plt.axis([0, 80, 0, 50])
     plt.show()


polynomial_regression = Pipeline([("poly_features", PolynomialFeatures(degree=8, include_bias=False)),("lin_reg", LinearRegression()),])
X_b=np.c_[Data_Group["age"]]
Y=Data_Group["income"]
plot_learning_curves(polynomial_regression,X_b,Y )





#Degree 15
poly_features=PolynomialFeatures(degree=15,include_bias=False)
X=T_Data["age"].to_numpy().T
Y=T_Data["income"].to_numpy().T
X_b=np.c_[X]
X_poly=poly_features.fit_transform(X_b)

lin_reg=LinearRegression()
lin_reg.fit(X_poly,Y)
print("Degree 15 Estimated Model Parameters",lin_reg.intercept_,lin_reg.coef_)
X_min=min(X)
X_max=max(X)
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_T=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Training Degree15 : MSE  "+str(MSE_T)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()
#validation Dataset
X=Val_Data["age"].to_numpy().T
Y=Val_Data["income"].to_numpy().T
X_b=np.c_[X]
X_new=X_b
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
MSE_val=mean_squared_error(Y,y_predict)
plt.plot(X, Y, "b.", linewidth=2, label="Real")
plt.plot(X_new, y_predict, "r.", linewidth=2, label="Predictions")
plt.xlabel("$Age$", fontsize=18)
plt.ylabel("$Income$", rotation=0, fontsize=18)
plt.legend(loc="upper left", fontsize=14)
plt.title(("Validation Degree15 : MSE  "+str(MSE_val)), rotation=0, fontsize=18)
plt.axis([X_min, X_max, 0, 500])
plt.show()


