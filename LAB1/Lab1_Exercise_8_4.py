#Lab1 Exercises
#Exercise 8.4
def any_lowercase1(s):
    for c in s:
        if c.islower():
            return True
        else:
            return False
# any_lowercase1(s) Function detects correctly that input contains any lowercase letters.
def any_lowercase2(s):
    for c in s:
        if "c".islower():
            return "True"
        else:
            return "False"
# any_lowercase2(s) Function doesnt detect correctly any lowercase letters. Becuase in head of if statement, c is not variable. it is string. Function is checking if c string is lowercase instead of input.

def any_lowercase3(s):
    for c in s:
        flag=c.islower()
    return flag

# any_lowercase3(s) Function detect if it is lowercase or not. However, it is not evaluating any lowercase in string. at the end of iteration,it is showing status of latest letter. if we have "atakaN" string, result  is no.

def any_lowercase4(s):
    flag=False
    for c in s:
        flag = flag or c.islower()
    return flag
# any_lowercase4(s) Function detects correctly that input contains any lowercase letters.
def any_lowercase5(s):

    for c in s:
        if not c.islower():
            return False
    return True
# any_lowercase5(s) Function doesnt detect correctly. if function detects capital, it directly return false without checking other letter.
