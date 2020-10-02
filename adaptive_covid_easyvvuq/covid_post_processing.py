"""
==============================================================================
THE POSTPROCESSING STEP

Analyse results after all sampling has completed
==============================================================================
"""
       
import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt
import numpy as np
import chaospy as cp

plt.close('all')

# home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
config = 'PC_CI_HQ_SD_suppress_campaign3_1_repeat'
# ID = ''
ID = '_campaign3_1'

#reload Campaign, sampler, analysis
campaign = uq.Campaign(state_file='covid_easyvvuq_state' + ID + '.json', work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
sampler.load_state('covid_sampler_state' + ID +'.pickle')
campaign.set_sampler(sampler)
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
analysis.load_state('covid_analysis_state' + ID + '.pickle')

#apply analysis
# campaign.apply_analysis(analysis)
# results = campaign.get_last_analysis()

data_frame = campaign.get_collation_result()
results = analysis.analyse(data_frame, compute_moments = True, compute_Sobols = True)

############################
# plot confidence interval #
############################

# n_samples = 500

# #draw n_samples draws from the input distributions
# xi_mc = np.zeros([n_samples, analysis.N])
# idx = 0
# for dist in sampler.vary.get_values():
#     xi_mc[:, idx] = dist.sample(n_samples)
#     idx += 1

# #sample the surrogate n_samples times
# surr_samples = np.zeros([n_samples, analysis.N_qoi])
# print('Sampling surrogate %d times' % (n_samples,))
# for i in range(n_samples):
#     surr_samples[i, :] = analysis.surrogate(output_columns[0], xi_mc[i])
#     if np.mod(i, 10) == 0:
#         print('%d of %d' % (i + 1, n_samples))
# print('done')

# # #confidence bounds
# lower, upper = analysis.get_confidence_intervals(output_columns[0], 500, surr_samples=surr_samples)

# #statistical moments
mean = results["statistical_moments"][output_columns[0]]["mean"]
std = results["statistical_moments"][output_columns[0]]["std"]

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="days", ylabel=output_columns[0])
ax.plot(mean, label=r'mean')
# ax.plot(lower, '--r', label='90% confidence')
# ax.plot(upper, '--r')
leg = plt.legend(loc=0)
leg.set_draggable(True)
ax.plot(mean + std, '--r')
ax.plot(mean - std, '--r')
plt.tight_layout()

#################################
# Plot some convergence metrics #
#################################

analysis.adaptation_histogram()
analysis.plot_stat_convergence()
surplus_errors = analysis.get_adaptation_errors()
fig = plt.figure()
ax = fig.add_subplot(111, xlabel = 'refinement step', ylabel='max surplus error')
ax.plot(range(1, len(surplus_errors) + 1), surplus_errors, '-b*')
plt.tight_layout()

##################################
# Plot first-order Sobol indices #
##################################

fig = plt.figure(figsize=[8, 4])
ax = fig.add_subplot(121, title=r'First-order Sobols indices',
                      xlabel="days", ylabel=output_columns[0], ylim=[0,1])
sobols_first = results["sobols_first"][output_columns[0]]
for param in sobols_first.keys():
    ax.plot(sobols_first[param], label=param)
leg = ax.legend(loc=0, fontsize=8)
leg.set_draggable(True)
plt.tight_layout()

#############################
# Uncertainty blowup number #
#############################

blowup = analysis.get_uncertainty_blowup(output_columns[0])

####################
# Adaptation table #
####################

params = list(sampler.vary.get_keys())
sobols = results['sobols_first']['cumDeath']
tmp = np.array([sobols[params[i]][-1] for i in range(analysis.N)]).flatten()
order = np.flipud(np.argsort(tmp))
analysis.adaptation_table(order=order)

plt.show()