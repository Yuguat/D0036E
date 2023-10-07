
import time

def has_duplicates(Value):
    if type(Value)!=list:
        print("input is not valid")
    else:
        V_Sorted=sorted(Value)

        for i in range(len(V_Sorted)-1):
            if V_Sorted[i] == V_Sorted[i+1]:
                return True

        return False

def histogram(s):
    d = dict()
    for c in s:
        if c not in d:
            d[c] = 1
        else:
            return True
    return d
Test=[1,2,3,4,4,10,30,40,50,60,70,40,5,8,10,21003102030]
start_time=float(time.time())
print(start_time)
print(histogram(Test))
end_time = float(time.time())
print(end_time)
execution_time = end_time - start_time
print(execution_time)
start_time=float(time.time())
print(start_time)
print(has_duplicates(Test))
end_time = float(time.time())
print(end_time)
execution_time2 = end_time - start_time
print(execution_time2)