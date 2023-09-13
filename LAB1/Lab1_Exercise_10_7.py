#Lab1 Exercises
#Exercise 10.7
def has_duplicates(Value):
    if type(Value)!=list:
        print("input is not valid")
    else:
        V_Sorted=sorted(Value)

        for i in range(len(V_Sorted)-1):
            if V_Sorted[i] == V_Sorted[i+1]:
                return True

        return False
