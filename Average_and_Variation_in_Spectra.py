#----------------------------------file notes---------------------------------

# Creates a combined plot of the average mapping spectra and also individual
# plots of each spectrum at each mapping point with the resultant average. 

#--------------------------------import modules-------------------------------
import numpy as np
import matplotlib.pyplot as plt
import time

#---------------------------------start timer---------------------------------
start_time = time.process_time()


#-------------------------------Data Processing-------------------------------

all_data = dict() # dictionary for concatinating all data
av_data = dict() # dictionary for storing averaged spectra intensity
for i in material:
    all_data[i] = np.zeros((1,4)) # initial placeholders
    f_sum = np.zeros(norm_data[i][xmin,ymin][:,3].shape) # initialise array
    #----standard----
    xmin = int(min(raw_data[i][:,0]))
    xmax = int(max(raw_data[i][:,0]) + xstep)
    ymin = int(min(raw_data[i][:,1]))
    ymax = int(max(raw_data[i][:,1]) + ystep)
    #----------------
    for x in range(xmin, xmax, xstep):
        for y in range(ymin, ymax, ystep):  
            spectrum = norm_data[i][x,y] # (1)
            all_data[i] = np.concatenate((all_data[i], spectrum)) # (1)
            f_sum = f_sum + norm_data[i][x,y][:,3] # (2)
    av_data[i] = f_sum/len(norm_data[i]) # (2)     
    all_data[i] = np.delete(all_data[i], 0, 0) # deletes initial placeholders

# (1) Takes all the seperate normalised data and and joins into one array
#     to allow for plotting all. If plotted each seperatley, gets messy.
# (2) Sums all spectra at their given indices then devides that sum by the
#     number of spectra, giving the average. 

#-----------------------------------Plotting----------------------------------


#-----data spread-----    
for i in material: 
    plt.figure(figsize=(7,5))
    xs = all_data[i][:,2]
    ys = all_data[i][:,3]
    plt.plot(xs, ys, ".", markersize = 1, color = "silver")
    xs = norm_data[i][xmin,ymin][:,2]
    av_ys = av_data[i]
    plt.plot(xs, av_ys, "k-", linewidth = 1, label = i + " average") 
    plt.axis([1200, 1750, 0.15, 1.3])
    plt.xticks(np.arange(1200, 1751, step = 50), rotation = 30)
    plt.yticks([],[])   
    plt.tick_params(axis ='x', direction ='in', which = "both")
    plt.minorticks_on()
    plt.xlabel("Raman shift (cm⁻¹)")
    plt.ylabel("Intensity (arb. units)")
    plt.legend(loc="upper right", fontsize="small", markerfirst=True,
           edgecolor="k", fancybox=False)
    # fig.tight_layout() will fit everything nicely
    plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\Figures\\" + i + " spread.pdf")


#-----all averages-----
fig = plt.figure(figsize=(7,5))
counter = 0
for i in material:
    #-----------------
    # differentiate line styles for different materials
    counter += 1
    if counter == 1:
        #line = "-"
        colour = "black"
        #markertype = "d"
    elif counter == 2:
        #line = "--"
        colour = "dimgrey"
        #markertype = "^"
    elif counter == 3:
        #line = ":"
        colour = "darkgray"
        #markertype = "*"
    elif counter == 4:
        #line = "-."
        colour = "lightgray"
        #markertype = "h"
    #-----------------
    xs = norm_data[i][xmin,ymin][:,2]
    av_ys = av_data[i]
    plt.plot(xs, av_ys, "-", label = i + " average", color = colour, \
             linewidth = 1.5) 
    plt.axis([1200, 1750, 0.3, 1.2])
    plt.xticks(np.arange(1200, 1751, step = 50), rotation = 30)
    plt.yticks([],[])   
    plt.tick_params(axis ='x', direction ='in', which = "both")
    plt.minorticks_on()
    plt.xlabel("Raman shift (cm⁻¹)")
    plt.ylabel("Intensity (arb. units)")
    plt.legend(loc="upper right", fontsize="small", markerfirst=True,
           edgecolor="k", fancybox=False)
plt.savefig("C:\\Users\\Hector\\Desktop\\Data\\Figures\\map averages.pdf")

#-----------------------------------------------------------------------------

# End process timer
end_time = time.process_time()
print("\nScript runtime: %.2f \bs" % (end_time - start_time))

# last runtime = 3.36s
# w/ savefig = 18.13s