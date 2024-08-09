
# plot fig2
from utils import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# read data
LBB_LoS = [90.906, 92.342, 93.420, 94.243, 94.888]
LBB_NLoS = [13.410, 17.179, 21.719, 26.926, 32.558]
CKM_LoS = [86.924, 88.855, 90.365, 91.551, 92.491]
CKM_NLoS = [76.026, 78.461, 80.472, 82.216, 83.788]
RM_LoS = [87.380, 89.383, 90.884, 92.028, 92.921]
RM_NLoS = [81.936, 84.501, 86.603, 88.279, 89.620]

fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,8))
plt.xlabel('SNR (dB)', fontsize=24)
plt.ylabel('Ratio (%)', fontsize=24)
index = np.arange(5)
plt.xticks(index, (10,15,20,25,30))

ax.plot(index, LBB_LoS, linewidth = 2, marker = '^', markersize=10,
        color = '#F25022', linestyle = '-', label='GBM LoS')
ax.plot(index, CKM_LoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFB900', linestyle = '-', label='CKM LoS')
ax.plot(index, RM_LoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7FBA00', linestyle = '-', label='Radio map LoS')
ax.plot(index, CKM_NLoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFB900', linestyle = '--', label='CKM NLoS')
ax.plot(index, RM_NLoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7FBA00', linestyle = '--', label='Radio map NLoS')
for label in ax.get_yticklabels():
    label.set_fontsize(24)
ax.set_ylim(73, 97)
ax.tick_params(bottom=False)

ax2.plot(index, LBB_LoS, linewidth = 2, marker = '^', markersize=10,
        color = '#F25022', linestyle = '-', label='GBM LoS')
ax2.plot(index, CKM_LoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFB900', linestyle = '-', label='CKM LoS')
ax2.plot(index, RM_LoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7FBA00', linestyle = '-', label='Radio map LoS')
ax2.plot(index, LBB_NLoS, linewidth = 2, marker = '^', markersize=10,
         color = '#F25022', linestyle = '--', label='GBM NLoS')
ax2.plot(index, CKM_NLoS, linewidth = 2, marker = 's', markersize=10,
        color = '#FFB900', linestyle = '--', label='CKM NLoS')
ax2.plot(index, RM_NLoS, linewidth = 2, marker = 'o', markersize=10,
        color = '#7FBA00', linestyle = '--', label='Radio map NLoS')

ax2.set_ylim(11, 36)
ax2.set_ylabel('', fontsize=24)
for label2 in ax2.get_xticklabels():
    label2.set_fontsize(24)
for label2 in ax2.get_yticklabels():
    label2.set_fontsize(24)

fig.text(0.04, 0.5, 'Ratio (%)', va='center', rotation='vertical', fontsize=24)
fig.subplots_adjust(hspace=0.05)
ax2.legend(fontsize=18, loc=(0.03,0.35), framealpha=1, facecolor='white')
ax.grid(True, ls=':', color='black', alpha=0.3)
ax2.grid(True, ls=':', color='black', alpha=0.3)

d = 0.01  # The length of the break mark
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1)
ax.plot((-d, +d), (-d, +d), **kwargs)
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax.grid(True, ls=':', color='black', alpha=0.3)
plt.savefig('picture/fig2.jpg')
plt.show()
