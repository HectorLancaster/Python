

# by putting "list" in the brackets when creating a new class, python is told
# to inherit the attributes of the object "list" into the new class. 
class MyList(list):
    def remove_min(self):
        self.remove(min(self))
    def remove_max(self):
        self.remove(max(self))

x = [10, 3, 5, 1, 2, 7, 6, 4, 8]
y = MyList(x)
dir(x)
dir(y)

print(y)

y.remove_min()
print(y)

y.remove_max()
print(y)
