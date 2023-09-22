
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def reading_csv(csv_file):
    f = open(csv_file, "r")
    age=[]
    income=[]

    for line in f.readlines()[1:]:
        V1=line.replace("\n","")
        V2=V1.split(",")
        age.append(float(V2[1]))
        income.append(float(V2[2]))


    a=list(range(0,(len(age))))
    d = {'age': age,'income': income}
    df=pd.DataFrame(data=d, index=a)

    print(df)
    return df

#Task2
def linear_regression(Data_F):
    X=Data_F["age"].to_numpy().T
    Y=Data_F["income"].to_numpy().T
    X_b=np.c_[np.ones((len(X),1)),X]
    t1 = X_b.T.dot(X_b)
    t2 = np.linalg.inv(t1)  # What does inv do?
    t3 = t2.dot(X_b.T)
    theta_best = t3.dot(Y)
    return theta_best

def Make_prediction(theta,X_new_l):
    X_new = np.array(X_new_l)
    len(X_new)
    X_new_b = np.c_[np.ones((len(X_new), 1)), X_new]
    # We dot product with theta to get the prediction
    y_predict = X_new_b.dot(theta)
    return X_new,y_predict

def Seperate_Data(Data_F):
    D=Data_F
    Validation_Data=D.sample(frac=0.2, random_state = 42)
    Training_Data=D.drop(Validation_Data.index)
    return Validation_Data, Training_Data

def MSE_Calculation(theta,Data):
    X_i=Data["age"].to_numpy().T
    Y_i=Data["income"].to_numpy().T
    X_new, Y_predict = Make_prediction(theta,X_i)
    MSE=sum((Y_i-Y_predict)**2)/len(X_i)
    return MSE


#Task1
csv_file="inc_subset.csv"
df=reading_csv(csv_file)

#Task2.1
Val_Data,T_Data=Seperate_Data(df)
theta=linear_regression(T_Data)
print(theta)
#Task 2.2
X_min=min(T_Data["age"].to_numpy().T)
X_max=max(T_Data["age"].to_numpy().T)
X_new=[X_min,X_max]
X_new,Y_predict = Make_prediction(theta,X_new)
MSE_V=MSE_Calculation(theta,T_Data)
plt.plot(X_new, Y_predict, "r-")
plt.plot(T_Data["age"], T_Data["income"], ".b")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.title(("MSE  "+str(MSE_V)), rotation=0, fontsize=18)
plt.axis([15, 55, 0, 500])
plt.show()

#Task2.3
X_min=min(Val_Data["age"].to_numpy().T)
X_max=max(Val_Data["age"].to_numpy().T)
X_new=[X_min,X_max]
X_new,Y_predict = Make_prediction(theta,X_new)
MSE_V=MSE_Calculation(theta,Val_Data)
plt.plot(X_new, Y_predict, "r-")
plt.plot(Val_Data["age"], Val_Data["income"], ".b")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.title(("MSE  "+str(MSE_V)), rotation=0, fontsize=18)
plt.axis([15, 55, 0, 500])
plt.show()
