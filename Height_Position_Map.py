#--------------------------------file notes-----------------------------------

# Creates a combined plot containing all map g peak postition wrt their xy 
# coordinates for each material.


#--------------------------------user inputs----------------------------------

# Interpolation: choose either "lanczos" or "none"
inter = "lanczos"
# Enter the grid size of the figure, here it is a one by three.
rows = 1
cols = len(material)
# Enter the map coordinates for the scalebar start and end
# Here we have a 21,21 array corresponding to 100x100 microns, the gap between
# two points is thus 4 microns, so by setting the gap to span 5 points, this 
# corresponds to a length of 20 microns.
end = 0, 19
start = 5, 19

#------------------------------import modules---------------------------------

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#----------------------------start process timer------------------------------

start_time = time.process_time()


#--------------------------------find range-----------------------------------

maxlimits = list()
minlimits = list()
for i in material:
    minlimits.append(min(max_loc_linear[i]))
    maxlimits.append(max(max_loc_linear[i]))
    minimum = min(minlimits)
    maximum = max(maxlimits)


#---------------------------------plot data-----------------------------------

fig = plt.figure(figsize=(9.75,3)) # width and height in inches

grid = ImageGrid(fig, 111, # the parent figure, three digit subplot pos
                 nrows_ncols=(rows,cols), # num of rows and columns
                 axes_pad=0.15, # spacing between subplots, in inches
                 cbar_location="right", # location of colorbar
                 cbar_mode="single", # a single/one colorbar
                 cbar_size="7%", # width of the colorbar
                 cbar_pad=0.15, # spacing between colorbar and plot
                 )


counter = -1 # initialise counter
for ax in grid: # for each axis in the image grid
    counter += 1
    data = material[counter] # links material name to axis
    #-----
    im = ax.imshow(max_loc[data], # data to show
                   cmap = 'afmhot', # colours used
                   interpolation=inter, # type of interpolation between points
                   vmin = minimum, # set colour min and max,
                   vmax = maximum) # this makes scale consistent across all
    #-----
    ax.set_title(data, loc = "left") # title location
    ax.set_xticks([],[]) # remove ticks and labels form axes
    ax.set_yticks([],[]) 
    #-----
    ax.annotate("", # empty text string as don't require text
                xy=(end), xycoords="data", # arrow end point
                xytext =(start), textcoords="data", # arrow start point
                arrowprops=dict(arrowstyle="-", # arrow style = line 
                                linewidth=2, color = "black"))
        
ax.cax.colorbar(im) # axes into which the colorbar is drawn, "im"
                    # links the colorbar to the last image in the above loop


plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\Figures\\height position map.pdf")



#-----------------------------end process timer-------------------------------

# End process timer
end_time = time.process_time()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 0.31s

#---------------------------------script end----------------------------------