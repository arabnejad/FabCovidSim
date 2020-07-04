"""
==============================================================================
THE POSTPROCESSING STEP 

Without link to the full data set - just reloads pre-generated results 
==============================================================================
"""
       
import easyvvuq as uq
import os
import matplotlib.pyplot as plt
import numpy as np
import chaospy as cp

plt.close('all')

# home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
ID = '_campaign3_1'

#create a dummpy sampler, then overwrite with the saved sampler state
sampler = uq.sampling.SCSampler(vary={'x1':cp.Uniform(0,1)})
sampler.load_state('covid_sampler_state' + ID +'.pickle')
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
analysis.load_state('covid_analysis_state'+ ID + '.pickle')

#apply analysis
# campaign.apply_analysis(analysis)
# results = campaign.get_last_analysis()
results = analysis.results

#########################
# plot mean +/- std dev #
#########################

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="days", ylabel=output_columns[0])
mean = results["statistical_moments"][output_columns[0]]["mean"]
std = results["statistical_moments"][output_columns[0]]["std"]
ax.plot(mean)
ax.plot(mean + std, '--r')
ax.plot(mean - std, '--r')
plt.tight_layout()

#################################
# Plot some convergence metrics #
#################################

analysis.adaptation_histogram()
analysis.plot_stat_convergence()

#####################################
# Plot the random surrogate samples #
#####################################

fig = plt.figure(figsize=[12, 4])
ax = fig.add_subplot(131, xlabel='days', ylabel=output_columns[0],
                     title='Surrogate samples')
#generate n_mc samples from the input distributions
n_mc = 20
xi_mc = np.zeros([n_mc,sampler.xi_d.shape[1]])
idx = 0
for dist in sampler.vary.get_values():
    xi_mc[:, idx] = dist.sample(n_mc)
    idx += 1
    
# evaluate the surrogate at these values
print('Evaluating surrogate model', n_mc, 'times')
for i in range(n_mc):
    ax.plot(analysis.surrogate(output_columns[0], xi_mc[i]), 'g')
print('done')

##################################
# Plot first-order Sobol indices #
##################################

ax = fig.add_subplot(132, title=r'First-order Sobols indices',
                      xlabel="days", ylabel=output_columns[0], ylim=[0,1])
sobols_first = results["sobols_first"][output_columns[0]]
for param in sobols_first.keys():
    ax.plot(sobols_first[param], label=param)
leg = ax.legend(loc=0, fontsize=8)
leg.set_draggable(True)
plt.tight_layout()

plt.show()