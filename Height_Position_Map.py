#--------------------------------file notes-----------------------------------

# Creates a combined plot containing all map g peak postition wrt their xy 
# coordinates for each material.

# To do:
#       add scale bar of 20 microns
#       work out how to scale the RHS plot to be equal to the others
#       work out how to scale teh size of the colorbar


#--------------------------------user inputs----------------------------------

# Interpolation: choose either "lanczos" or "none"
inter = "lanczos"

#------------------------------import modules---------------------------------

import matplotlib.pyplot as plt

#----------------------------start process timer------------------------------

start_time = time.process_time()


#---------------------------------plot data-----------------------------------

counter = 0
plt.figure() 
for i in material:
    counter += 1
    plt.subplot(int("13" + str(counter)))
    plt.imshow(max_loc[i], aspect='equal', cmap='afmhot', interpolation=inter)
    plt.yticks([],[])
    plt.xticks([],[])
    plt.title(i, loc = "left")
    
plt.colorbar(fraction = 0.05)

plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\Figures\\height position map.pdf")


#-----------------------------end process timer-------------------------------

# End process timer
end_time = time.process_time()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 0.14s

#---------------------------------script end----------------------------------