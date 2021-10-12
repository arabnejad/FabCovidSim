import os
import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import easyvvuq as uq
from scipy import linalg
from custom import CustomEncoder

plt.close('all')
plt.rcParams['image.cmap'] = 'seismic'

home = os.path.abspath(os.path.dirname(__file__))

output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns/DAS'
ID = '_DAS1'

#reload Campaign, sampler, analysis
campaign = uq.Campaign(state_file="states/covid_easyvvuq_state" + ID + ".json", 
                       work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
sampler.load_state("states/covid_sampler_state" + ID + ".pickle")
campaign.set_sampler(sampler)

# load the data frame and turn it into a array of training data
data_frame = campaign.get_collation_result()
qoi_col = "cumDeath"
samples = []
run_id_int = [int(run_id.split('Run_')[-1]) for run_id in data_frame[qoi_col].keys()]
for run_id in range(1, np.max(run_id_int) + 1):
        samples.append(data_frame[qoi_col]['Run_' + str(run_id)])
# samples = np.array(samples)[:, -1]
samples = np.array(samples)
samples = samples/np.mean(samples[:, -1])
samples = samples[:, -1].reshape([-1, 1])
params = np.array(sampler.xi_d)

# number of inputs for a(x) and kappa(x)
D = params.shape[1]

surrogate = es.methods.ANN_Surrogate()
surrogate.train(params, samples, n_iter=10000, n_layers=4, n_neurons=50, test_frac = 0.1, 
                batch_size = 64)
dims = surrogate.get_dimensions()
#########################
# Compute error metrics #
#########################

# run the trained model forward at training locations
n_mc = dims['n_train']
pred = np.zeros([n_mc, dims['n_out']])
for i in range(n_mc):
    pred[i,:] = surrogate.predict(params[i])
   
train_data = samples[0:dims['n_train']]
rel_err_train = np.linalg.norm(train_data - pred) / np.linalg.norm(train_data)
print("Training error = %.4f" % rel_err_train)

# run the trained model forward at test locations
pred = np.zeros([dims['n_test'], dims['n_out']])
for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    pred[idx] = surrogate.predict(params[i])
test_data = samples[dims['n_train']:]
rel_err_test = np.linalg.norm(test_data - pred) / np.linalg.norm(test_data)
print("Test error = %.4f" % rel_err_test)

# n_mc = 10**4
# params = np.array([p.sample(n_mc) for p in sampler.vary.get_values()]).T
n_mc = params.shape[0]

surrogate.neural_net.set_batch_size(1)

C = 0.0
n_mc_samples = np.zeros([n_mc])

for i, param in enumerate(params):
    df_dx = surrogate.neural_net.d_norm_y_dX(param.reshape([1, -1]))
    n_mc_samples[i] = surrogate.predict(param)
    C += np.dot(df_dx, df_dx.T) / n_mc
    
eigvals, eigvecs = linalg.eigh(C)

# Sort the eigensolutions in the descending order of eigenvalues
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

d = 6
W1 = eigvecs[:, 0:d]
y = np.dot(W1.T, params.T).T

fig = plt.figure()
ax = fig.add_subplot(111)
ct = ax.tricontourf(y[:, 0], y[:, 1], n_mc_samples, 100)
plt.colorbar(ct)
plt.tight_layout()

idx = np.flipud(np.argsort(np.abs(W1[:,0])))
print(np.array(list(sampler.vary.get_keys()))[idx])

#########################

das_surrogate = es.methods.DAS_Surrogate()
das_surrogate.train(params, samples, d, n_iter=10000, n_layers=4, n_neurons=100, test_frac = 0.1, 
                batch_size = 64)

W1_das = das_surrogate.neural_net.layers[1].W

n_mc = params.shape[0]

das_surrogate.neural_net.set_batch_size(1)

C = 0.0
n_mc_samples = np.zeros([n_mc])
foo = 0.0

for i, param in enumerate(params):
    n_mc_samples[i] = das_surrogate.predict(param)
    df_dx = das_surrogate.neural_net.d_norm_y_dX(param.reshape([1, -1]))
    df_dh = das_surrogate.neural_net.layers[1].delta_hy.reshape([-1,1])
    foo += np.dot(df_dh, df_dh.T) / n_mc
    C += np.dot(df_dx, df_dx.T) / n_mc
    
eigvals, eigvecs = linalg.eigh(C)

# Sort the eigensolutions in the descending order of eigenvalues
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

fig = plt.figure()
ax = fig.add_subplot(111, title=r'$d=%d$' % d)
ax.set_ylabel(r'$\lambda_i$', fontsize=12)
ax.set_xlabel(r'$i$', fontsize=12)
ax.set_xticks(np.arange(1, D+1))
ax.plot(np.arange(1, D+1), eigvals, 'ro')
ax.set_yscale('log')
plt.tight_layout()

fig = plt.figure(figsize=[8,4])
ax = fig.add_subplot(121, title='AS')
W1 = eigvecs[:, 0:d]
y = np.dot(W1.T, params.T).T
ct = ax.tricontourf(y[:, 0], y[:, 1], n_mc_samples, 100)
ax.set_xlabel(r'$y_1$')
ax.set_ylabel(r'$y_2$')
plt.colorbar(ct)
#
ax = fig.add_subplot(122, title='DAS')
y_das = np.dot(W1_das.T, params.T).T
ct = ax.tricontourf(y_das[:, 0], y_das[:, 1], n_mc_samples, 100)
ax.set_xlabel(r'$y_1$')
ax.set_ylabel(r'$y_2$')
plt.colorbar(ct)
plt.tight_layout()

# R = np.dot(W1.T, W1_das)
# y_trans = np.dot(R, y.T).T

# ax = fig.add_subplot(133, title='TRANS')
# ct = ax.tricontourf(y_trans[:, 0], y_trans[:, 1], n_mc_samples, 100)
# plt.colorbar(ct)
# plt.tight_layout()

eigvals_red, eigvecs_red = linalg.eigh(foo)
order = eigvals_red.argsort()[::-1]
eigvals_red = eigvals_red[order]
eigvecs_red = eigvecs_red[:, order]

print(np.dot( W1_das, np.dot(foo, W1_das.T)) - C)
print(np.dot(eigvecs_red.T, np.dot(foo, eigvecs_red)) - np.diag(eigvals[0:d]))
print(np.dot(W1, np.dot(np.dot(eigvecs_red.T, np.dot(foo, eigvecs_red)), W1.T)) - C)

correct_sign = np.array(np.sign([np.dot(np.dot(W1, eigvecs_red.T[:,i]), W1_das[:,i]) for i in range(d)]))
eigvecs_red *= correct_sign
# print(W1_das - np.dot(W1, eigvecs_red.T))
# print(correct_sign)
print(np.abs(eigvecs_red.T) - np.abs(np.dot(W1.T, W1_das)))

# A1 = np.copy(eigvecs_red)
# A2 = np.copy(eigvecs_red)
# A3 = np.copy(eigvecs_red)
# A4 = np.copy(eigvecs_red)

# A2[:, 0] *= -1
# A3[:, 1] *= -1
# A4 *= -1

# print(W1_das - np.dot(W1, A1.T))
# print(W1_das - np.dot(W1, A2.T))
# print(W1_das - np.dot(W1, A3.T))
# print(W1_das - np.dot(W1, A4.T))
