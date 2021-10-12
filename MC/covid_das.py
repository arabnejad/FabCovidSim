def plot_3D_convex_hull(points, ax):

    hull = ConvexHull(points)
    
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]

        surf = ax.plot_trisurf(np.array([p1[0], p2[0], p3[0]]), 
                        np.array([p1[1], p2[1], p3[1]]),
                        np.array([p1[2], p2[2], p3[2]]), 
                        alpha=0.5, color='coral',
                        label=r'Convex hull active subspace')
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d
        
def get_confidence_intervals(n_samples, conf=0.9, 
                             method='bootstrap', **kwargs):
    #ake sure conf is in [0, 1]
    if conf < 0.0 or conf > 1.0:
        print('conf must be specified within [0, 1]')
        return
    #lower bound = alpha, upper bound = 1 - alpha
    alpha = 0.5*(1.0 - conf)

    #use precomputed surrogate samples or not
    if 'surr_samples' in kwargs:
        surr_samples = kwargs['surr_samples']

    #arrays for lower and upper bound of the interval
    lower = np.zeros(N_qoi)
    upper = np.zeros(N_qoi)

    #the probabilities of the ecdf
    prob = np.linspace(0, 1, n_samples)
    #the closest locations in prob that correspond to the interval bounds
    idx0 = np.where(prob <= alpha)[0][-1]
    idx1 = np.where(prob <= 1.0 - alpha)[0][-1]

    #for every location of qoi compute the ecdf-based confidence interval
    for i in range(N_qoi):
        #the sorted surrogate samples at the current location
        samples_sorted = np.sort(surr_samples[:, i])
        #the corresponding confidence interval
        lower[i] = samples_sorted[idx0]
        upper[i] = samples_sorted[idx1]

    return lower, upper

import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import easyvvuq as uq
from custom import CustomEncoder
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import seaborn as sns

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
samples = samples[:, 100:-1:10]
params = np.array(sampler.xi_d)

#save part of the data for testing
test_frac = 0.25
I = np.int(samples.shape[0] * (1.0 - test_frac))

#train a DAS network
D = 19
d = 7

surrogate = es.methods.DAS_Surrogate()
surrogate.train(params, samples, d, n_iter=20000, n_layers=4, n_neurons=100, test_frac = 0.0,
                batch_size=128)
# surrogate = es.methods.ANN_Surrogate()
# surrogate.train(params, samples, n_iter=10000, n_layers=5, n_neurons=100, test_frac = 0.2)

dims = surrogate.get_dimensions()

#########################
# Compute error metrics #
#########################

# run the trained model forward at training locations
n_mc = dims['n_train']
pred1 = np.zeros([n_mc, dims['n_out']])
for i in range(n_mc):
    pred1[i,:] = surrogate.predict(params[i])
   
train_data = samples[0:dims['n_train']]
rel_err_train = np.linalg.norm(train_data - pred1)/np.linalg.norm(train_data)

# run the trained model forward at test locations
pred2 = np.zeros([dims['n_test'], dims['n_out']])
for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    pred2[idx] = surrogate.predict(params[i])
test_data = samples[dims['n_train']:]
rel_err_test = np.linalg.norm(test_data - pred2)/np.linalg.norm(test_data)

print('================================')
print('Relative error on training set = %.4f' % rel_err_train)
print('Relative error on test set = %.4f' % rel_err_test)
print('================================')

#################################
# Plot the confidence intervals #
#################################

#perform some basic analysis
analysis = es.analysis.DAS_analysis(surrogate)

N_qoi = samples.shape[1]
x = range(N_qoi)

#confidence bounds
lower1, upper1 = analysis.get_confidence_intervals(pred1, conf=0.63)
lower2, upper2 = analysis.get_confidence_intervals(pred1, conf=0.95)

fig = plt.figure(figsize=(10,5))
spec = gridspec.GridSpec(ncols=2, nrows=1,
                          width_ratios=[3, 1])

ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1], sharey=ax1)
ax2.get_xaxis().set_ticks([])
fig.subplots_adjust(wspace=0)
plt.setp(ax2.get_yticklabels(), visible=False)

ax1.fill_between(x, lower2, upper2, color='#aa99cc', label='95% CI', alpha=0.5)
ax1.fill_between(x, lower1, upper1, color='#aa99cc', label='68% CI')

mean = np.mean(pred1, axis=0)
ax1.plot(x, mean, label='Mean')

ax1.legend(loc="upper left")

ax1.set_xlabel('Days')
ax1.set_ylabel('Cumulative deaths')
ax2.axis('off')

total_deaths = pred1[:, -1]
sns.histplot(y=total_deaths, ax=ax2)

plt.tight_layout()

##############################
# plot PDF final death count #
##############################

dom_ref, pdf_ref = analysis.get_pdf(samples[:, -1], 100)
dom_das, pdf_das = analysis.get_pdf(pred1[:, -1], 100)

fig = plt.figure()
ax = fig.add_subplot(111, yticks=[], xlabel=r'$f(x), g(y)$')
ax.plot(dom_ref, pdf_ref, '--', label='reference')
ax.plot(dom_das, pdf_das, label='deep active subspace')
leg = ax.legend(loc=0)
leg.set_draggable(True)
plt.tight_layout()

###########################
# Sensitivity experiments #
###########################

# n_mc = 10**4
# params = np.array([p.sample(n_mc) for p in sampler.vary.get_values()]).T
idx, mean = analysis.sensitivity_measures(params)
params_ordered = np.array(list(sampler.vary.get_keys()))[idx[0]]

fig = plt.figure('sensitivity', figsize=[4, 8])
ax = fig.add_subplot(111)
ax.set_ylabel(r'$\int\frac{\partial ||y||^2_2}{\partial x_i}p({\bf x})d{\bf x}$', fontsize=14)
# find max quad order for every parameter
ax.bar(range(mean.size), height = mean[idx].flatten())
ax.set_xticks(range(mean.size))
ax.set_xticklabels(params_ordered)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# if samples.ndim == 2:
#     n_out = samples.shape[1]
# else:
#     n_out = 1

# surrogate = es.methods.DAS_network(params[0:I], samples[0:I], d, n_layers=4, 
#                                     n_out=n_out, n_neurons=100, batch_size=128,
#                                     decay_rate=0.9, decay_step = 10**4,  
#                                     activation='tanh',
#                                     activation_out='linear', save=False,
#                                     cumsum=False, standardize_y=True)

# # surrogate = es.methods.ANN(params[0:I], samples[0:I], n_layers=4, 
# #                                     n_out=n_out, n_neurons=100, batch_size=128,
# #                                     decay_rate=0.9, decay_step = 10**4,  
# #                                     activation='hard_tanh', save=False)

# # train the surrogate on the data
# n_iter = 20000
# surrogate.train(n_iter, store_loss=True)

# # run the trained model forward at training locations
# n_mc = samples.shape[0]
# pred = np.zeros([I, n_out])
# y = np.zeros([I, d])
# for i in range(I):
#     pred[i,:] = surrogate.feed_forward(surrogate.X[i].reshape([1, -1])).flatten()
#     y[i,:] = surrogate.layers[1].h[0:d].flatten()
# if surrogate.y.ndim == 2:
#     data = surrogate.y[:, -1].reshape([-1, 1])
# else:
#     data = surrogate.y.reshape([-1, 1])
    
# # #plot the active subspace
# # plt.close('all')
# # if d == 1:
# #     fig = plt.figure()
# #     ax = plt.subplot(111, xlabel=r'${\bf y} = W^T\xi$',
# #                      title='Link function g(y) in active subspace')
# #     ax.plot(y, data[:, -1], '+', label='data')
# #     ax.plot(y, pred[:, -1], 's', label='prediction g(y)', alpha=0.5)
# #     plt.legend()

# # else:
# #     fig = plt.figure(figsize=[8, 4])
# #     ax1 = fig.add_subplot(121, projection='3d',
# #                           xlabel=r'$y_1$', ylabel=r'$y_2$',
# #                           title='Link function g(y) in active subspace')
# #     if d == 2:
# #         ax1.plot_trisurf(y[:, 0], y[:, 1], pred[:, -1], alpha=0.5)
# #     else:
# #         points = np.array([y[:, 0], y[:, 1], pred[:, -1]]).T
# #         plot_3D_convex_hull(points, ax1)
# #     ax1.plot(y[:, 0], y[:, 1], data[:, 0], 'o', markersize=3, label=r'data')
# #     handles, labels = ax1.get_legend_handles_labels()
# #     by_label = dict(zip(labels, handles))
# #     plt.legend(by_label.values(), by_label.keys())

# #     ax2 = fig.add_subplot(122, xlabel=r'$y_1$', ylabel=r'$y_2$',
# #                           title='Sampling plan in active subspace')
# #     ax2.plot(y[:, 0], y[:, 1], 'o')

# # plt.tight_layout()

# # #################################
# # # Plot the confidence intervals #
# # #################################

# from matplotlib import gridspec
# import seaborn as sns

# N_qoi = samples.shape[1]
# x = range(N_qoi)

# # code_samples = pred * surrogate.y_std + surrogate.y_mean
# code_samples = samples 

# n_samples = code_samples.shape[0]

# #confidence bounds
# lower1, upper1 = get_confidence_intervals(n_samples, conf=0.63, surr_samples=code_samples)
# lower2, upper2 = get_confidence_intervals(n_samples, conf=0.95, surr_samples=code_samples)

# fig = plt.figure(figsize=(10,5))
# spec = gridspec.GridSpec(ncols=2, nrows=1,
#                           width_ratios=[3, 1])

# ax1 = fig.add_subplot(spec[0], xlim=[0, N_qoi], ylim=[0, 2])
# ax2 = fig.add_subplot(spec[1], sharey=ax1)
# ax2.get_xaxis().set_ticks([])
# fig.subplots_adjust(wspace=0)
# plt.setp(ax2.get_yticklabels(), visible=False)

# ax1.fill_between(x, lower2, upper2, color='#aa99cc', label='95% CI', alpha=0.5)
# ax1.fill_between(x, lower1, upper1, color='#aa99cc', label='68% CI')

# mean = np.mean(code_samples, axis=0)
# ax1.plot(x, mean, label='Mean')

# # median = np.median(code_samples, axis=0)
# # ax1.plot(x, median, label='Median')

# ax1.legend(loc="upper left")

# ax1.set_xlabel('Days')
# ax1.set_ylabel('Cumulative deaths')
# # ax2.set_xlabel('Frequency')
# #ax2.set_title('Total deaths distribution')
# ax2.axis('off')

# total_deaths = code_samples[:, -1]
# ax2 = sns.distplot(total_deaths, vertical=True)

# plt.tight_layout()

# # Other analysis

# #perform some basic analysis
# analysis = es.analysis.BaseAnalysis()
# dom_ref, pdf_ref = analysis.get_pdf(data, 100)
# # pred = pred*surrogate.y_std + surrogate.y_mean
# dom_das, pdf_das = analysis.get_pdf(pred[:, -1], 100)

# fig = plt.figure()
# ax = fig.add_subplot(111, yticks=[], xlabel=r'$f(x), g(y)$')
# ax.plot(dom_ref, pdf_ref, '--', label='reference')
# ax.plot(dom_das, pdf_das, label='deep active subspace')
# leg = ax.legend(loc=0)
# leg.set_draggable(True)
# plt.tight_layout()

# fig = plt.figure()
# plt.plot((pred * surrogate.y_std).T + surrogate.y_mean.reshape([-1,1]))

# # Compute error metrics
# print('================================')
# pred = np.zeros([I, n_out])
# for i in range(I):
#     feat_i = (params[i] - surrogate.X_mean) / surrogate.X_std
#     pred_i = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
#     pred[i] = pred_i * surrogate.y_std + surrogate.y_mean
    
# train_data = samples[0:I].reshape([I, -1])
# rel_err_train = np.linalg.norm(train_data - pred)/np.linalg.norm(train_data)

# print('Relative error on training set = %.4f' % rel_err_train)

# pred = np.zeros([n_mc - I, n_out])
# idx = 0
# for i in range(I, n_mc):
#     feat_i = (params[i] - surrogate.X_mean) / surrogate.X_std
#     pred_i = surrogate.feed_forward(feat_i.reshape([1, -1]))[0][0]
#     pred[idx] = pred_i * surrogate.y_std + surrogate.y_mean
#     idx += 1
# test_data = samples[I:].reshape([n_mc - I, -1])
# rel_err_test = np.linalg.norm(test_data - pred)/np.linalg.norm(test_data)
# print('Relative error on test set = %.4f' % rel_err_test)
# print('================================')

# # Sensitivity experiments

# if d == 1:
#     inputs = np.array(list(sampler.vary.get_keys()))
#     idx = np.argsort(np.abs(surrogate.layers[1].W.T))
#     print('Parameters ordered from most to least important:')
#     print(np.fliplr((inputs[idx])))
    
# surrogate.set_batch_size(1)
# surrogate.d_norm_y_dX(surrogate.X[0].reshape([1,-1]))
# f_grad_y = surrogate.layers[1].delta_hy
# f_grad_x = surrogate.layers[0].delta_hy
# mean_f_grad_y2 = f_grad_y**2
# mean_f_grad_x2 = f_grad_x**2
# var = np.zeros(d)
# var2 = np.zeros(D)
# analysis = es.analysis.BaseAnalysis()

# for i in range(1, surrogate.X.shape[0]):
#     surrogate.d_norm_y_dX(surrogate.X[i].reshape([1,-1]))
#     f_grad_y2 = surrogate.layers[1].delta_hy**2
#     f_grad_x2 = surrogate.layers[0].delta_hy**2
#     mean_f_grad_y2, var = analysis.recursive_moments(f_grad_y2, mean_f_grad_y2, var, i)
#     mean_f_grad_x2, var2 = analysis.recursive_moments(f_grad_x2, mean_f_grad_x2, var2, i)

# inputs = np.array(list(sampler.vary.get_keys()))
# idx = np.argsort(np.abs(mean_f_grad_x2).T)
# print('Parameters ordered from most to least important:')
# print(np.fliplr((inputs[idx])))