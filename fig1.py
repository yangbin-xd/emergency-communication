
# plot fig4
from utils import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# read data
part32 = [92.552, 96.584, 98.576, 99.331, 99.633]
pilot32 = [93.032, 96.808, 98.615, 99.304, 99.577]
pilot24 = [91.689, 95.849, 97.595, 98.500, 98.892]
pilot16 = [90.880, 94.484, 96.268, 97.196, 97.737]
pilot8 = [84.001, 87.686, 89.940, 91.350, 92.493]

# calculate practical data rate
part32 = np.array(part32)   * (1-0.0952)
pilot32 = np.array(pilot32) * (1-0.0952)
pilot24 = np.array(pilot24) * (1-0.0714)
pilot16 = np.array(pilot16) * (1-0.0476)
pilot8 = np.array(pilot8)   * (1-0.0238)

fig,ax = plt.subplots(figsize=(10,8))
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('Ratio (%)', fontsize=24)
index = np.arange(len(part32)) + 1
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xticks(index, (10,15,20,25,30))

plt.plot(index, pilot32, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '-', label=r'$\mathcal{\eta}=$'+'100% trained')
plt.plot(index, pilot24, linewidth = 2, marker = 's', markersize=10,
         color = '#FFB900', linestyle = '-', label=r'$\mathcal{\eta}=$'+'75% trained')
plt.plot(index, pilot16, linewidth = 2, marker = 'o', markersize=10,
         color = '#7FBA00', linestyle = '-', label=r'$\mathcal{\eta}=$'+'50% trained')
plt.plot(index, pilot8, linewidth = 2, marker = 'd', markersize=10,
         color = '#00A4EF', linestyle = '-', label=r'$\mathcal{\eta}=$'+'25% trained')
plt.plot(index, part32, linewidth = 2, marker = 'p', markersize=10,
         color = '#7030A0', linestyle = '-', label=r'$\mathcal{\eta}=$'+'100% untrained')
plt.ylim([81, 94])

plt.legend(loc = 'lower right', fontsize=18)
ax.grid(True, ls=':', color='black', alpha=0.3)
plt.savefig('picture/fig1.jpg')
plt.show()
