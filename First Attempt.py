
with open("\\Users\\Hector\\Desktop\\Data\\Glass Slide.txt", "r") as glass_slide:
              test = glass_slide.readlines()

spectrum = [line.split() for line in test]
wavenumber = [float(spectrum[i][0]) for i in range(len(spectrum))]
intensity = [float(spectrum[i][1]) for i in range(len(spectrum))]

norm_intensity = [intensity[i]/max(intensity) for i in range(len(intensity))]

import matplotlib.pyplot as plt

# Plot data, label="First" gives a label to the dataset in the plot
plt.plot(wavenumber, norm_intensity)

#plt.xticks([],[])
#plt.yticks([],[])

# LaTeX is used to alter phonts, if "$X$" then the X is in italics
# This labels the y and x axes
plt.xlabel("Wavenumber")
plt.ylabel("Intensity")

# plt.axis([xmin,xmax,ymin,ymax]) sets the bounds of the axis
# plt.axis([-0.5, 10.5, -5, 105])

# This is where teh legend will be displayed
# plt.legend(loc="upper left")

# This is the name of the file, save location and file type,
# the directory can be input here, remember \\ escape sequence
plt.savefig("glass slide.pdf")