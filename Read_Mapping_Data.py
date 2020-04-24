
#----------------------------Import Data--------------------------------------
with open("\\Users\\Hector\\Desktop\\Data\\KC10 map.txt", "r") as KC10:
              KC10_raw = KC10.readlines()
with open("\\Users\\Hector\\Desktop\\Data\\LiC10 map.txt", "r") as LiC10:
              LiC10_raw = LiC10.readlines()
with open("\\Users\\Hector\\Desktop\\Data\\yp50 map.txt", "r") as yp50:
              yp50_raw = yp50.readlines()

#-----------------------------Sort Data---------------------------------------
spectrum = [line.split() for line in KC10_raw]
length = len(spectrum) 
x_KC10 = [int(spectrum[i][0]) for i in range(length)]
y_KC10 = [int(spectrum[i][1]) for i in range(length)]
wavenumber_KC10 = [float(spectrum[i][2]) for i in range(length)]
intensity_KC10 = [float(spectrum[i][3]) for i in range(length)]

data_points = x_KC10.count(0)
spectra_KC10 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x_KC10[i] == a and y_KC10[i] == b:
                data.append((wavenumber_KC10[i], intensity_KC10[i]))
        spectra_KC10[a,b] = data


spectrum = [line.split() for line in LiC10_raw]
length = len(spectrum) 
x_LiC10 = [int(spectrum[i][0]) for i in range(length)]
y_LiC10 = [int(spectrum[i][1]) for i in range(length)]
wavenumber_LiC10 = [float(spectrum[i][2]) for i in range(length)]
intensity_LiC10 = [float(spectrum[i][3]) for i in range(length)]

data_points = x_LiC10.count(0)
spectra_LiC10 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x_LiC10[i] == a and y_LiC10[i] == b:
                data.append((wavenumber_LiC10[i], intensity_LiC10[i]))
        spectra_LiC10[a,b] = data

    
spectrum = [line.split() for line in yp50_raw]
length = len(spectrum) 
x_yp50 = [int(spectrum[i][0]) for i in range(length)]
y_yp50 = [int(spectrum[i][1]) for i in range(length)]
wavenumber_yp50 = [float(spectrum[i][2]) for i in range(length)]
intensity_yp50 = [float(spectrum[i][3]) for i in range(length)]
        
data_points = x_yp50.count(0)
spectra_yp50 = dict()
for a in range(0,105,5):            
    for b in range(-100,5,5):
        data = list()
        for i in range(length):
            if x_yp50[i] == a and y_yp50[i] == b:
                data.append((wavenumber_yp50[i], intensity_yp50[i]))
        spectra_yp50[a,b] = data