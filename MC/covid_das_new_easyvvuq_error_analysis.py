import numpy as np
import matplotlib.pyplot as plt
import easysurrogate as es
import easyvvuq as uq
from custom_new import CustomEncoder
# from scipy.spatial import ConvexHull
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
import seaborn as sns

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
# samples = samples['cumDeath'][:,200:-1:10]
samples = samples['cumDeath'][:,-1].reshape([-1, 1])

#train a DAS network
D = 41
n_d = 15
d = np.arange(1, n_d)

train_err = []
test_err = []

for i in range(d.size+1):
    
    if i < d.size:
        surrogate = es.methods.DAS_Surrogate()
        surrogate.train(params, samples, d[i], n_iter=10000, 
                        n_layers=4, n_neurons=100, test_frac = 0.2)
    else:
        surrogate = es.methods.ANN_Surrogate()
        surrogate.train(params, samples, n_iter=10000, 
                        n_layers=4, n_neurons=100, test_frac = 0.2)
        
    dims = surrogate.get_dimensions()
    
    #########################
    # Compute error metrics #
    #########################
    
    # run the trained model forward at training locations
    n_mc = dims['n_train']
    pred = np.zeros([n_mc, dims['n_out']])
    for j in range(n_mc):
        pred[j,:] = surrogate.predict(params[j])
       
    train_data = samples[0:dims['n_train']]
    rel_err_train = np.linalg.norm(train_data - pred)/np.linalg.norm(train_data)
    
    # run the trained model forward at test locations
    pred = np.zeros([dims['n_test'], dims['n_out']])
    for idx, j in enumerate(range(dims['n_train'], dims['n_samples'])):
        pred[idx] = surrogate.predict(params[j])
    test_data = samples[dims['n_train']:]
    rel_err_test = np.linalg.norm(test_data - pred)/np.linalg.norm(test_data)
    
    print('================================')
    print('Relative error on training set = %.4f' % rel_err_train)
    print('Relative error on test set = %.4f' % rel_err_test)
    print('================================')
    train_err.append(rel_err_train)
    test_err.append(rel_err_test)

fig = plt.figure(figsize=[4,4])
ax = fig.add_subplot(111)
ax.plot(d, train_err[0:-1], '-rs', label='training error')
ax.plot(d, test_err[0:-1], '-b*', label='test error')
ax.plot([d[0], d[-1]], np.ones(2) * train_err[-1], '--k', label=r'ANN train error')
ax.plot([d[0], d[-1]], np.ones(2) * test_err[-1], '-k', label=r'ANN test error')
leg = plt.legend(loc=0)
plt.xlabel('d')
plt.ylabel('relative error')
plt.tight_layout()
ax.set_xticks(d)
plt.show()
