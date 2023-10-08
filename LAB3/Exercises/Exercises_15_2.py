import turtle
import math


def draw_rectangle(bob):
    Rect_Height = input("Height of Rectangle")
    Rect_Length = input("Length of Rectangle")
    Height=int(Rect_Height)
    Length=int(Rect_Length)
    bob.fd(Length)
    bob.lt(90)
    bob.fd(Height)
    bob.lt(90)
    bob.fd(Length)
    bob.lt(90)
    bob.fd(Height)

def polyline(t, n, length, angle):
    for i in range(n):
        t.fd(length)
        t.lt(angle)


def arc(t, r, angle):
    arc_length = 2 * math.pi * r * angle / 360

    n = int(arc_length / 3) + 1
    step_length = arc_length / n
    step_angle = float(angle) / n
    polyline(t, n, step_length, step_angle)

def circle(t):
    Radius = input("Radius of Rectangle")
    r = int(Radius)
    arc(t, r, 360)





bob = turtle.Turtle()
draw_rectangle(bob)
circle(bob)
turtle.mainloop()