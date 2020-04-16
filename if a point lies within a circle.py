def distance(x, y):
    """
    distance(x,y) gives the distance between two points: x = [a1, b1] and 
    y = [a2, b2]
    """
    import math
    diff = math.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
    print(diff) 



def in_circle(x, origin = [0,0]):
    import math
    diff = math.sqrt((0-x[0])**2 + (0-x[1])**2)
    print(diff < 1)
    
in_circle([1,1])