"""
==============================================================================
THE ADAPTATION STEP

Execute the look ahead step first

Takes the admissble points computed in the look ahead step to decide along 
which dimension to place more samples. See adapt_dimension subroutine.

The look-ahead step and the adaptation step can be executed multiple times:
look ahead, adapt, look ahead, adapt, etc
==============================================================================
"""
       
import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

# home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
config = 'PC_CI_HQ_SD_suppress_campaign3_1_repeat'

#reload Campaign, sampler, analysis
campaign = uq.Campaign(state_file="covid_easyvvuq_state.json", 
                       work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
sampler.load_state("covid_sampler_state.pickle")
campaign.set_sampler(sampler)
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
analysis.load_state("covid_analysis_state.pickle")

fab.get_uq_samples(config, campaign.campaign_dir, sampler._number_of_samples,
                   machine='eagle_vecma')
campaign.collate()

#compute the error at all admissible points, select direction with
#highest error and add that direction to the grid

data_frame = campaign.get_collation_result()
analysis.adapt_dimension(output_columns[0], data_frame)

#save everything
campaign.save_state("covid_easyvvuq_state.json")
sampler.save_state("covid_sampler_state.pickle")
analysis.save_state("covid_analysis_state.pickle")

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

# #########################
# # plot mean +/- std dev #
# #########################

# #apply analysis
# campaign.apply_analysis(analysis)
# results = campaign.get_last_analysis()

# fig = plt.figure()
# ax = fig.add_subplot(111, xlabel="days", ylabel=output_columns[0])
# mean = results["statistical_moments"][output_columns[0]]["mean"]
# std = results["statistical_moments"][output_columns[0]]["std"]
# ax.plot(mean)
# ax.plot(mean + std, '--r')
# ax.plot(mean - std, '--r')
# plt.tight_layout()

# #####################################
# # Plot the random surrogate samples #
# #####################################

# fig = plt.figure(figsize=[12, 4])
# ax = fig.add_subplot(131, xlabel='days', ylabel=output_columns[0],
#                       title='Surrogate samples')
# ax.plot(analysis.get_sample_array(output_columns[0]).T, 'ro', alpha = 0.5)

# #generate n_mc samples from the input distributions
# n_mc = 20
# xi_mc = np.zeros([n_mc,sampler.xi_d.shape[1]])
# idx = 0
# for dist in sampler.vary.get_values():
#     xi_mc[:, idx] = dist.sample(n_mc)
#     idx += 1
# xi_mc = sampler.xi_d
# n_mc = sampler.xi_d.shape[0]
    
# # evaluate the surrogate at these values
# print('Evaluating surrogate model', n_mc, 'times')
# for i in range(n_mc):
#     ax.plot(analysis.surrogate(output_columns[0], xi_mc[i]), 'g')
# print('done')

# ##################################
# # Plot first-order Sobol indices #
# ##################################

# ax = fig.add_subplot(132, title=r'First-order Sobols indices',
#                       xlabel="days", ylabel=output_columns[0], ylim=[0,1])
# sobols_first = results["sobols_first"][output_columns[0]]
# for param in sobols_first.keys():
#     ax.plot(sobols_first[param], label=param)
# leg = ax.legend(loc=0, fontsize=8)
# leg.set_draggable(True)
# plt.tight_layout()

# plt.show()