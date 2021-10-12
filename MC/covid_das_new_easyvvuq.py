import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import easyvvuq as uq
from custom_new import CustomEncoder
# from scipy.spatial import ConvexHull
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import seaborn as sns
from custom import CustomEncoder

output_columns = ["cumDeath"]
WORK_DIR = '/home/wouter/VECMA/Campaigns/DAS_41'
# WORK_DIR = '/tmp'
# FabSim3 config name 
CONFIG = 'PC_CI_HQ_SD_DAS_all_param'
# Simulation identifier
ID = '_DAS_41'
# EasyVVUQ campaign name
CAMPAIGN_NAME = CONFIG + ID
# location of the EasyVVUQ database
DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID

###################
# reload Campaign #
###################
campaign = uq.Campaign(name=CAMPAIGN_NAME, db_location=DB_LOCATION)
print("===========================================")
print("Reloaded campaign {}".format(CAMPAIGN_NAME))
print("===========================================")

sampler = campaign.get_active_sampler()
campaign.set_sampler(sampler, update=True)

surr_campaign = es.Campaign()
params, samples = surr_campaign.load_easyvvuq_data(campaign, output_columns)
samples = samples['cumDeath'][:,200:-1:10]

#train a DAS network
D = 41
d = 5

surrogate = es.methods.DAS_Surrogate()
surrogate.train(params, samples, d, n_iter=10000, n_layers=4, n_neurons=100, test_frac = 0.2)
# surrogate = es.methods.ANN_Surrogate()
# surrogate.train(params, samples, n_iter=10000, n_layers=5, n_neurons=100, test_frac = 0.2)

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
rel_err_train = np.linalg.norm(train_data - pred)/np.linalg.norm(train_data)

# run the trained model forward at test locations
pred = np.zeros([dims['n_test'], dims['n_out']])
for idx, i in enumerate(range(dims['n_train'], dims['n_samples'])):
    pred[idx] = surrogate.predict(params[i])
test_data = samples[dims['n_train']:]
rel_err_test = np.linalg.norm(test_data - pred)/np.linalg.norm(test_data)

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
lower1, upper1 = analysis.get_confidence_intervals(pred, conf=0.63)
lower2, upper2 = analysis.get_confidence_intervals(pred, conf=0.95)

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

mean = np.mean(pred, axis=0)
ax1.plot(x, mean, label='Mean')

ax1.legend(loc="upper left")

ax1.set_xlabel('Days')
ax1.set_ylabel('Cumulative deaths')
ax2.axis('off')

total_deaths = pred[:, -1]
sns.histplot(y=total_deaths, ax=ax2)

plt.tight_layout()

##############################
# plot PDF final death count #
##############################

dom_ref, pdf_ref = analysis.get_pdf(samples[:, -1], 100)
dom_das, pdf_das = analysis.get_pdf(pred[:, -1], 100)

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

n_mc = 10**4
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
