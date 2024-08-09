
# Integrate radio map and pilot output
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

# resource block
CSI_RB = CSI[:,:,None,:,:] # (N, Nc, 1, Nr, Nt)
Ns = 14 # 5G NR RB
CSI_RB = np.repeat(CSI_RB, Ns, axis=2) # (N, Nc, Ns, Nr, Nt)
CSI_RB = CSI_RB * (1/np.sqrt(2) + 1j/np.sqrt(2))
noise_real = np.random.normal(0, np.sqrt(noise/2), (N, Nc, Ns, Nr, Nt))
noise_imag = np.random.normal(0, np.sqrt(noise/2), (N, Nc, Ns, Nr, Nt))
noise_comp = noise_real + 1j * noise_imag
CSI_RB = CSI_RB + noise_comp
CSI_RB = CSI_RB / (1/np.sqrt(2) + 1j/np.sqrt(2))

# interpolation
pilot_num = 32
CSI_part = inter(CSI_RB, pilot_num)
CSI_part = np.mean(CSI_part, axis=2)
U, Sigma, VT = np.linalg.svd(CSI_part)
reduced_pilot = VT[:,:,0,:].conjugate()

# shuffle
seed = 1
np.random.seed(seed)
np.random.shuffle(loc_norm)
np.random.seed(seed)
np.random.shuffle(reduced_pilot)
np.random.seed(seed)
np.random.shuffle(CSI_noise)
np.random.seed(seed)
np.random.shuffle(CSI)
np.save('data/CSI_noise.npy', CSI_noise)

# ratio
ratio = 0.8
N_train = int(N * ratio)
N_test = N - N_train

# compute v_RM
RM_model = models.load_model('model/radio_map.h5',
                          custom_objects={'cust_loss': su_loss(noise)})
v_RM = RM_model.predict(loc_norm)
v_RM = normalize_v(v_RM)
np.save(f'data/v_RM{SNR}.npy', v_RM)
v_RM_test = RM_model.predict(loc_norm[N_train:,:])
v_RM_test = normalize_v(v_RM_test)

# compute v_pilot
pilot_model = models.load_model(f'model/pilot{pilot_num}.h5',
                          custom_objects={'cust_loss': su_loss(noise)})
x = np.empty([N, Nc, Nt, 2])
x[:,:,:,0] = np.real(reduced_pilot)
x[:,:,:,1] = np.imag(reduced_pilot)
v_pilot = pilot_model.predict(x)
v_pilot = normalize_v(v_pilot)
np.save(f'data/v_pilot{SNR}.npy', v_pilot)
v_pilot_test = pilot_model.predict(x[N_train:,:])
v_pilot_test = normalize_v(v_pilot_test)

# (x_train, y_train, x_test, y_test)
x = np.concatenate([np.real(v_RM), np.imag(v_RM),
                    np.real(v_pilot), np.imag(v_pilot)], axis=-1)
x_train = x[0:N_train,:]
x_test = x[N_train:,:]
y_train = CSI_noise[0:N_train,:]
y_test = CSI_noise[N_train:,:]

# build model
input = layers.Input(shape=(Nc,Nt,4))
output = fusion(input)
model = models.Model(inputs=[input], outputs=[output])

# # train
# model, history = train(x_train, y_train, x_test, y_test, model, 200, noise)
# model.save(f'model/fusion{SNR}.h5')
# history_dict = history.history
# for key in history_dict:
#     history_dict[key] = [float(i) for i in history_dict[key]]
# with open(f'loss/fusion{SNR}.json', 'w') as f:
#     json.dump(history_dict, f)

# test
model = models.load_model(f'model/fusion{SNR}.h5',
                          custom_objects={'cust_loss': su_loss(noise)})
y_fusion = model.predict(x)
v_fusion = normalize_v(y_fusion) # (N_test, Nc, Nt, 1)
np.save(f'data/v_fusion{SNR}.npy', v_fusion)
y_pred = model.predict(x_test)
v_norm = normalize_v(y_pred) # (N_test, Nc, Nt, 1)

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI[N_train:,:])
v_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI[N_train:,:], v_opt, noise)

# calculate spectral efficiency
RM_SE = cal_SE(CSI[N_train:,:], v_RM_test, noise)
pilot_SE = cal_SE(CSI[N_train:,:], v_pilot_test, noise) * (1-0.0952)
fusion_SE = cal_SE(CSI[N_train:,:], v_norm, noise) * (1-0.0952)

RM2opt = np.mean(RM_SE) / np.mean(opt_SE) * 100
pilot2opt = np.mean(pilot_SE) / np.mean(opt_SE) * 100
fusion2opt = np.mean(fusion_SE) / np.mean(opt_SE) * 100

# print results
print('RM_SE:', np.round(RM2opt, 3), '%')
print('pilot_SE:',np.round(pilot2opt, 3), '%')
print('fusion_SE:', np.round(fusion2opt, 3), '%')
