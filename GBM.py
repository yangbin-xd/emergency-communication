
# location based beamforming
from utils import *

# read Data
BSloc = np.load('data/BSloc.npy')
UEloc = np.load('data/UEloc.npy')
CSI = np.load('data/CSI.npy')

# eliminate block points
[UEloc, CSI] = elimi_block(UEloc, CSI)

# process
BS = np.round(BSloc[0,0:2], 3)
loc = UEloc[:,0:2]
CSI = np.transpose(CSI, [0,3,1,2])
[N, Nc, Nr, Nt] = (CSI.shape)

# shuffle
np.random.seed(1)
np.random.shuffle(loc) # (N, 2)
np.random.seed(1)
np.random.shuffle(CSI) # (N, Nc, Nr, Nt)

# training ratio
a = 0.8
N_train = int(N * a)
N_test = N - N_train
loc_train = loc[0:N_train,:]
loc_test = loc[N_train:,:]
CSI_train = CSI[0:N_train,:]
CSI_test = CSI[N_train:,:]
np.save('data/CSI_test.npy', CSI_test)

# calculate AoD based on locations
AoD = np.empty(N_test)
for i in np.arange(N_test):
    AoD[i] = math.atan((loc_test[i,1] - BS[1]) / 
                       (loc_test[i,0] - BS[0])) * 180 / np.pi
    
# communication parameters
c = 3e8
f = 3.5e9
lamda = c/f
B = 18e6
d = 1/2*lamda

# calculate transmit precoding vector
v_LBB = np.empty([N_test, Nc, Nt, 1], dtype=complex)
for i in range(N_test):
    AoD[i] = AoD[i] * np.pi / 180
    for j in range(Nc):
        lamda = c / (f + j*B/Nc)
        for n in range(Nt):
            v_LBB[i,j,n] = np.exp(-1j*2*np.pi*n*d*np.sin(AoD[i])/lamda)

# output normalization
v_LBB = normalize_v(v_LBB)
np.save('data/LBB.npy', v_LBB)

# noise
SNR = 30
power = (np.linalg.norm(CSI))**2/N/Nc/Nr/Nt
noise = power / (10**(SNR/10))

# calculate SE of LBB
LBB_SE = cal_SE(CSI_test, v_LBB, noise)
print('LBB SE:', np.mean(LBB_SE))

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI_test)
v_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI_test, v_opt, noise)
print('opt_SE:', np.mean(opt_SE))

# calculate ratio
LBB2opt = np.mean(LBB_SE) / np.mean(opt_SE) * 100
print('LBB2opt:', np.round(LBB2opt, 3), '%')

# calculate LoS and NLoS conditions
LBB_LoS_SE, LBB_NLoS_SE = compute(LBB_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)

LBB2opt_LoS = np.mean(LBB_LoS_SE) / np.mean(opt_LoS_SE) * 100
LBB2opt_NLoS = np.mean(LBB_NLoS_SE) / np.mean(opt_NLoS_SE) * 100

print('LBB LoS:', np.round(LBB2opt_LoS, 3), '%')
print('LBB NLoS:', np.round(LBB2opt_NLoS, 3), '%')
