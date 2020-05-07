# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:54:06 2020

@author: Hector
"""

amp1 = 1
wid1 = 7
cen1 = 1593

y_array_lorentz = histogram[0]
y_length = int(y_array_gauss.shape[0])
x_array = np.zeros(y_length,)
for i in range(y_length):
    x_array[i] = (histogram[1][i]+histogram[1][i+1])/2


fig = plt.figure(figsize=(4,3))
gs = gridspec.GridSpec(1,1)
ax1 = fig.add_subplot(gs[0])

ax1.plot(x_array, y_array_lorentz, "ro")

#ax1.set_xlim(-5,105)
#ax1.set_ylim(-0.5,5)

ax1.set_xlabel("x_array",family="serif",  fontsize=12)
ax1.set_ylabel("y_array",family="serif",  fontsize=12)

#ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
#ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

#ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
#ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

ax1.tick_params(axis='both',which='major', direction="out", top="on", right="on", bottom="on", length=8, labelsize=8)
ax1.tick_params(axis='both',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()

def _1Lorentzian(x, amp1, cen1, wid1):
    return (amp1*wid1**2/((x-cen1)**2+wid1**2))

popt_lorentz, pcov_lorentz = scipy.optimize.curve_fit(_1Lorentzian, x_array, y_array_lorentz, p0=[amp1, cen1, wid1])

perr_lorentz = np.sqrt(np.diag(pcov_lorentz))


pars_1 = popt_lorentz[0:3]
lorentz_peak_1 = _1Lorentzian(x_array, *pars_1)

print("-------------Peak 1-------------")
print("amplitude = %0.2f (+/-) %0.2f" % (pars_1[0], perr_lorentz[0]))
print("center = %0.2f (+/-) %0.2f" % (pars_1[1], perr_lorentz[1]))
print("width = %0.2f (+/-) %0.2f" % (pars_1[2], perr_lorentz[2]))
print("area = %0.2f" % np.trapz(lorentz_peak_1))

residual_lorentz = y_array_lorentz - (_1Lorentzian(x_array, *popt_lorentz))


fig = plt.figure(figsize=(4,4))
gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
gs.update(hspace=0) 

ax1.plot(x_array, y_array_lorentz, "ro")
ax1.plot(x_array, _1Lorentzian(x_array, *popt_lorentz), 'k--')#,\
         #label="y= %0.2f$e^{%0.2fx}$ + %0.2f" % (popt_exponential[0], popt_exponential[1], popt_exponential[2]))

# peak 1
ax1.plot(x_array, lorentz_peak_1, "g")
ax1.fill_between(x_array, lorentz_peak_1.min(), lorentz_peak_1, facecolor="green", alpha=0.5)

ax2.set_xlabel("x_array",family="serif",  fontsize=12)
ax1.set_ylabel("y_array",family="serif",  fontsize=12)
ax2.set_ylabel("Res.",family="serif",  fontsize=12)

ax1.legend(loc="best")

ax1.tick_params(axis='x',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='x',which='minor', direction="out", top="on", right="on", bottom="off", length=5, labelsize=8)
ax1.tick_params(axis='y',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
ax1.tick_params(axis='y',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

ax2.tick_params(axis='x',which='major', direction="out", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='x',which='minor', direction="out", top="off", right="on", bottom="on", length=5, labelsize=8)
ax2.tick_params(axis='y',which='major', direction="out", top="off", right="on", bottom="on", length=8, labelsize=8)
ax2.tick_params(axis='y',which='minor', direction="out", top="off", right="on", bottom="on", length=5, labelsize=8)

fig.tight_layout()