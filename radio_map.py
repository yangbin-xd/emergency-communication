
# radio map scheme
from utils import *

# read data
UEloc = np.load('data/UEloc.npy') # (2000, 3)
CSI = np.load('data/CSI.npy') # (2000, 1, 32, 12)

# eliminate block points
[UEloc, CSI] = elimi_block(UEloc, CSI)

# process
CSI = CSI.transpose(0,3,1,2)
[N, Nc, Nr, Nt] = CSI.shape
loc = UEloc[:, 0:2]
loc_norm = normalize_loc(loc)

# add noise
SNR = 0
power = (np.linalg.norm(CSI))**2/N/Nc/Nr/Nt
noise = power / (10**(SNR/10))
CSI_pilot = CSI * (1/np.sqrt(2) + 1j/np.sqrt(2))
noise_real = np.random.normal(0, np.sqrt(noise/2), (N, Nc, Nr, Nt))
noise_imag = np.random.normal(0, np.sqrt(noise/2), (N, Nc, Nr, Nt))
noise_comp = noise_real + 1j * noise_imag
CSI_pilot = CSI_pilot + noise_comp
CSI_noise = CSI_pilot / (1/np.sqrt(2) + 1j/np.sqrt(2))
CSI_noise = CSI_noise.astype(np.complex64)

# shuffle
seed = 1
np.random.seed(seed)
np.random.shuffle(loc_norm)
np.random.seed(seed)
np.random.shuffle(CSI_noise)
np.random.seed(seed)
np.random.shuffle(CSI)

# (x_train, y_train, x_test, y_test)
ratio = 0.8
N_train = int(N * ratio)
N_test = N - N_train
x_train = loc_norm[0:N_train,:]
x_test = loc_norm[N_train:,:]
y_train = CSI_noise[0:N_train,:]
y_test = CSI_noise[N_train:,:]

# build model
input = layers.Input(shape=2)
output = radio_map_model(input)
model = models.Model(inputs=[input], outputs=[output])

# # train
# model, history = train(x_train, y_train, x_test, y_test, model, 1000, noise)
# model.save('model/radio_map.h5')
# history_dict = history.history
# for key in history_dict:
#     history_dict[key] = [float(i) for i in history_dict[key]]
# with open('loss/radio_map.json', 'w') as f:
#     json.dump(history_dict, f)

# test
model = models.load_model('model/radio_map.h5',
                          custom_objects={'cust_loss': su_loss(noise)})

# calculate spectral efficiency
v_pred = model.predict(x_test)
v_norm = normalize_v(v_pred)
radio_map_SE = cal_SE(CSI[N_train:,:], v_norm, noise)
print('radio map SE:', np.mean(radio_map_SE))

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI[N_train:,:])
v_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI[N_train:,:], v_opt, noise)
print('opt SE:', np.mean(opt_SE))

# calculate ratio
radio_map2opt = np.mean(radio_map_SE) / np.mean(opt_SE) * 100
print('radio map2opt:', np.round(radio_map2opt, 3), '%')

# calculate LoS and NLoS conditions
RM_LoS_SE, RM_NLoS_SE = compute(radio_map_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)

RM2opt_LoS = np.mean(RM_LoS_SE) / np.mean(opt_LoS_SE) * 100
RM2opt_NLoS = np.mean(RM_NLoS_SE) / np.mean(opt_NLoS_SE) * 100

print('radio map LoS:', np.round(RM2opt_LoS, 3), '%')
print('radio map NLoS:', np.round(RM2opt_NLoS, 3), '%')