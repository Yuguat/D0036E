#Lab1 Exercises
#Exercise 10.6
def is_anagram(V1,V2):
    if V1!= V2 and (sorted(V1) == sorted(V2)):
        return True
    else:
        return False


