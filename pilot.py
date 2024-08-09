
# pilot to precoding vector
from utils import *

# read data
UEloc = np.load('data/UEloc.npy') # (2000, 3)
CSI = np.load('data/CSI.npy') # (2000, 1, 32, 12)

# eliminate block points
[UEloc, CSI] = elimi_block(UEloc, CSI)

# process
CSI = CSI.transpose(0,3,1,2)
[N, Nc, Nr, Nt] = CSI.shape

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
pilot_num = 8
CSI_part = inter(CSI_RB, pilot_num)
CSI_part = np.mean(CSI_part, axis=2)
U, Sigma, VT = np.linalg.svd(CSI_part)
reduced_pilot = VT[:,:,0,:].conjugate()

# shuffle
seed = 1
np.random.seed(seed)
np.random.shuffle(reduced_pilot)
np.random.seed(seed)
np.random.shuffle(CSI_noise)
np.random.seed(seed)
np.random.shuffle(CSI)

# (x_train, y_train, x_test, y_test)
ratio = 0.8
N_train = int(N * ratio)
N_test = N - N_train
x = np.empty([N, Nc, Nt, 2])
x[:,:,:,0] = np.real(reduced_pilot)
x[:,:,:,1] = np.imag(reduced_pilot)
x_train = x[0:N_train,:]
x_test = x[N_train:,:]
y_train = CSI_noise[0:N_train,:]
y_test = CSI_noise[N_train:,:]

# build model
input = layers.Input(shape=(Nc,Nt,2))
output = pilot(input)
model = models.Model(inputs=[input], outputs=[output])

# # train
# model, history = train(x_train, y_train, x_test, y_test, model, 100, noise)
# model.save(f'model/pilot{pilot_num}.h5')
# history_dict = history.history
# for key in history_dict:
#     history_dict[key] = [float(i) for i in history_dict[key]]
# with open(f'loss/pilot{pilot_num}.json', 'w') as f:
#     json.dump(history_dict, f)

# test
model = models.load_model(f'model/pilot{pilot_num}.h5',
                          custom_objects={'cust_loss': su_loss(noise)})

# calculate spectral efficiency
v_pred = reduced_pilot[:,:,:,None][N_train:,:]  # (N_test, Nc, Nt, 1)
part_SE = cal_SE(CSI[N_train:,:], v_pred, noise)
# print('part_SE:', np.mean(part_SE))

y_pred = model.predict(x_test)
v_norm = normalize_v(y_pred) # (N_test, Nc, Nt, 1)
pilot_SE = cal_SE(CSI[N_train:,:], v_norm, noise)
# print('pilot_SE:', np.mean(pilot_SE))

# calculate optimum
U, Sigma, VT = np.linalg.svd(CSI[N_train:,:])
v_opt = VT[:,:,0,:][:,:,:,None].conjugate()
opt_SE = cal_SE(CSI[N_train:,:], v_opt, noise)
# print('opt_SE:', np.mean(opt_SE))

# calculate ratio
part2opt = np.mean(part_SE) / np.mean(opt_SE) * 100
print('part2opt:', np.round(part2opt, 3), '%')

pilot2part = np.mean(part_SE) / np.mean(pilot_SE) * 100
print('part2pilot:', np.round(pilot2part, 3), '%')

pilot2opt = np.mean(pilot_SE) / np.mean(opt_SE) * 100
print('pilot2opt:', np.round(pilot2opt, 3), '%')

# calculate LoS and NLoS conditions
pilot_LoS_SE, pilot_NLoS_SE = compute(pilot_SE)
opt_LoS_SE, opt_NLoS_SE = compute(opt_SE)

pilot2opt_LoS = np.mean(pilot_LoS_SE) / np.mean(opt_LoS_SE) * 100
pilot2opt_NLoS = np.mean(pilot_NLoS_SE) / np.mean(opt_NLoS_SE) * 100

print('pilot LoS:', np.round(pilot2opt_LoS, 3), '%')
print('pilot NLoS:', np.round(pilot2opt_NLoS, 3), '%')