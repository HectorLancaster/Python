import random

random.seed(1) # fixes the seed of the random number generator.

def rand():
    random_number = random.uniform(-1,1)
    print(random_number)

rand()