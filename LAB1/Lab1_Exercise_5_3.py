#Lab1 Exercises
#Exercise 5.3
def is_triangle(X,Y,Z):


    T_Sticks=sorted([X,Y,Z])
    if T_Sticks[0]<=0:
        print("Please give valid input")

    else:
        if sum(T_Sticks[0:2])<T_Sticks[2]:
            print("No")
        else:
            print("Yes")


#X,Y,Z= input("Enter your child age:").split()
X,Y,Z = [int(x) for x in input("Enter 3 sticks length: ").split()]
is_triangle(X,Y,Z)