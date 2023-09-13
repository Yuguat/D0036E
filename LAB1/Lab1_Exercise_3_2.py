#Lab1 Exercises
#Exercise 3.2.1
def do_twice_test(f):
    f()
    f()
def print_spam():
    print('spam')

print("Exercise 3.2.1")
do_twice_test(print_spam)

#Exercise 3.2.2
def do_twice(f,V):
    f(V)
    f(V)

#Exercise 3.2.3

def print_twice(V):
    print(V)
    print(V)

#Exercise 3.2.4
print("Exercise 3.2.4")
do_twice(print_twice,"Spam")

#Exercise 3.2.5

def do_four(f, V):
    do_twice(f, V)
    do_twice(f, V)
print("Exercise 3.2.5")
do_four(print, 'spam')