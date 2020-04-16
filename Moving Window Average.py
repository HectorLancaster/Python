def moving_window_average(x, n_neighbours):
    n = len(x)
    new_list = list()
    width = n_neighbours*2 + 1 
    x = [x[0]]*n_neighbours + x + [x[-1]]*n_neighbours
    for i in range(n):
        i += 1
        av = (x[i-1]+x[i]+x[i+1])/width   
        new_list = new_list + [av]
    print(sum(new_list))
    return new_list
    
    
x = [0,10,5,3,1,5]
print(moving_window_average(x,1))

# This is how you would do it using list comprehensions. Much better!

#def moving_window_average(x, n_neighbors=1):
    #n = len(x)
    #width = n_neighbors*2 + 1
    #x = [x[0]]*n_neighbors + x + [x[-1]]*n_neighbors
    #return [sum(x[i:(i+width)]) / width for i in range(n)]

#x = [0,10,5,3,1,5]
#print(sum(moving_window_average(x, 1)))
#print(moving_window_average(x, 1))