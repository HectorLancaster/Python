import random
random.seed(1)
R = 1000


x = [random.uniform(0,1) for i in range(R)]

    
def moving_window_average(x, n_neighbours):
    n = len(x)
    width = n_neighbours*2 + 1
    x_new = [x[0]]*n_neighbours + x + [x[-1]]*n_neighbours
    return [sum(x_new[i:(i+width)]) / width for i in range(n)]

Y = [x] + [moving_window_average(x,i) for i in range(1,10)]
print(Y[5][9])