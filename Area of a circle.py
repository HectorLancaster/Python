

def rand():
    import random
    return random.uniform(-1,1)

def distance(x, y):
    import math
    return math.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
    
def in_circle(x, origin = [0,0]):
    if len(x) != 2:
        return "x is not two-dimensional!"
    elif distance(x, origin) < 1:
        return True
    else:
        return False

R = 1000
inside = list()
T = int()
F = int()
for i in range(R):   
    x = [rand(),rand()]
    inside.append(in_circle(x))
print(sum(inside)/R)

    