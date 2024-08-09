
# show DeepMIMO O1_3p5 scenario
from utils import *

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'times new roman'

# read data
BSloc = np.load('data/BSloc.npy') # (1, 3)
UEloc = np.load('data/UEloc.npy') # (2000, 3)
LoS = np.load('data/LoS.npy') # (2000, )
CSI = np.load('data/CSI.npy') # (2000, 2, 32, 12)

# # eliminate block points
# [UEloc, CSI] = elimi_block(UEloc, CSI)
# n = UEloc.shape[0]

# # shuffle
# seed = 1
# np.random.seed(seed)
# np.random.shuffle(UEloc)
# UEloc = UEloc[int(n*0.4):int(n*0.8),:]

# process
CSI = CSI.transpose(0,3,1,2) # (2000, 12, 2, 32)
[N, Nc, Nr, Nt] = CSI.shape

'''
(1): The LoS path exists.
(0): Only NLoS paths exist. The LoS path is blocked (LoS blockage).
(-1): No paths exist between the transmitter and the receiver (Full blockage).

'''
LoS_index = [i for i, x in enumerate(LoS) if x==1]
NLoS_index = [i for i, x in enumerate(LoS) if x==0]
Block_index = [i for i, x in enumerate(LoS) if x==-1]

print('LoS number:', len(LoS_index))
print('NLoS number:', len(NLoS_index))
print('Block number:', len(Block_index))

# show scenario
fig,ax = plt.subplots(figsize=(14,4))
ax.scatter(BSloc[0,1], BSloc[0,0], marker='^', c='#F25022', s = 50, label='BS 3')
ax.scatter(UEloc[LoS_index,1], UEloc[LoS_index,0], marker='.',
           c='#7FBA00', s = 50, label='LoS')
ax.scatter(UEloc[NLoS_index,1], UEloc[NLoS_index,0], marker='.',
           c='#FFB900', s = 50, label='NLos')
ax.scatter(UEloc[Block_index,1], UEloc[Block_index,0], marker='.',
           c='#00A4EF', s = 50, label='Block')
ax.set_aspect(1)
# ax.invert_xaxis()
plt.legend(loc='upper right', fontsize=18, handletextpad=0.2, handlelength=1)
plt.xlabel('X (m)', fontsize=18)
plt.ylabel('Y (m)', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
x_major_locator=MultipleLocator(25)
y_major_locator=MultipleLocator(10)
plt.xlim(395, 595)
# plt.title('LoS map')
# plt.savefig('picture/scenario.jpg')
plt.show()