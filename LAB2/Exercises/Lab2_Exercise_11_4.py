import LAB1.Lab1_Exercise_10_7 as at
import time
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
print(histogram(Test))
end_time = float(time.time())
execution_time = end_time - start_time
print(execution_time)
start_time=float(time.time())
print(at.has_duplicates(Test))
end_time = float(time.time())
execution_time2 = end_time - start_time
print(execution_time2)