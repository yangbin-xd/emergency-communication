
# plot fig6
from utils import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

RM = [85.012, 87.184, 88.906, 90.262, 91.344]
pilot16 = [86.699, 90.189, 91.698, 92.639, 93.087]
fusion16 = [88.834, 91.617, 92.912, 93.773, 94.187]

fig,ax = plt.subplots(figsize=(10,8))
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('Ratio (%)', fontsize=24)
index = np.arange(len(RM)) + 1
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xticks(index, (10,15,20,25,30))

plt.plot(index, pilot16, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '-', label='Reduced pilots with '+r'$\eta=50\%$')
plt.plot(index, RM, linewidth = 2, marker = 's', markersize=10,
         color = '#FFB900', linestyle = '-', label='Radio map')
plt.plot(index, fusion16, linewidth = 2, marker = 'o', markersize=10,
         color = '#7FBA00', linestyle = '-', label='Integration')
plt.ylim([84, 95])

plt.legend(loc = 'lower right', fontsize=18)
ax.grid(True, ls=':', color='black', alpha=0.3)
ax.grid(True, ls=':', color='black', alpha=0.3)
plt.savefig('picture/fig3.jpg')
plt.show()
