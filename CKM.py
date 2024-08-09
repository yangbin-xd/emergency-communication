
# channel knowledge map
from utils import *

# read Data
UEloc = np.load('data/UEloc.npy')
CSI = np.load('data/CSI.npy')
AoD = np.load('data/AoD.npy')

# eliminate block points
[UEloc, CSI] = elimi_block(UEloc, CSI)
AoD = AoD[AoD != 0]

# preprocess
loc = UEloc[:,0:2]
CSI = np.transpose(CSI, [0,3,1,2])
[N, Nc, Nr, Nt] = (CSI.shape)

# shuffle
np.random.seed(1)
np.random.shuffle(loc) # (N, 2)
np.random.seed(1)
np.random.shuffle(CSI) # (N, Nc, Nr, Nt)
np.random.seed(1)
np.random.shuffle(AoD) # (N, )

# training ratio
ratio = 0.8
N_train = int(N * ratio)
N_test = N - N_train
loc_train = loc[0:N_train,:]
loc_test = loc[N-N_test:N,:]
CSI_train = CSI[0:N_train,:]
CSI_test = CSI[N_train:,:]
AoD_train = AoD[0:N_train]
AoD_test = AoD[N_train:]

# calculate distance to apply inverse distance weighting (IDW)
dist = np.empty([N_test, N_train])
for i in range(N_test):
    for j in range(N_train):
        dist[i,j] = np.sqrt(np.sum((loc_test[i] - loc_train[j]) ** 2))

# K nearest neighbors
def KNN(i, AoD_train, k):
    dist_sort = sorted(enumerate(dist[i,:]), key=lambda x:x[1])
    index = [x[0] for x in dist_sort]
    K = index[0:k]
    dist_k = dist[i,K]
    AoD_k = np.squeeze(AoD_train[K])
    weight = 1/dist_k
    weight = weight/np.sum(weight)
    AoD_IDW = np.dot(weight, AoD_k)
    return AoD_IDW

# communication parameters
c = 3e8
f = 3.5e9
lamda = c/f
B = 0.18e6
d = 1/2*lamda

# calculate transmit precoding vector
v_CKM = np.empty([N_test, Nc, Nt, 1], dtype=complex)
for i in range(N_test):
    theta = KNN(i, AoD_train, 3) * np.pi / 180
    for j in range(Nc):
        lamda = c / (f + j*B/Nc)
        for n in range(Nt):
            v_CKM[i,j,n] = np.exp(-1j*2*np.pi*n*d*np.sin(theta)/lamda)

# output normalization
v_CKM = normalize_v(v_CKM)

# noise
SNR = 30
power = (np.linalg.norm(CSI))**2/N/Nc/Nr/Nt
noise = power / (10**(SNR/10))

# calculate SE of CKM
CKM_SE = cal_SE(CSI_test, v_CKM, noise)
print('CKM SE:', np.mean(CKM_SE))

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI_test)
v_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI_test, v_opt, noise)
print('opt_SE:', np.mean(opt_SE))

# calculate ratio
CKM2opt = np.mean(CKM_SE) / np.mean(opt_SE) * 100
print('CKM2opt:', np.round(CKM2opt, 3), '%')

# calculate LoS and NLoS conditions
CKM_LoS_SE, CKM_NLoS_SE = compute(CKM_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)

CKM2opt_LoS = CKM_LoS_SE / opt_LoS_SE * 100
CKM2opt_NLoS = CKM_NLoS_SE / opt_NLoS_SE * 100

print('CKM LoS:', np.round(CKM2opt_LoS, 3), '%')
print('CKM NLoS:', np.round(CKM2opt_NLoS, 3), '%')
