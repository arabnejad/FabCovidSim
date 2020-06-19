"""
==============================================================================
ANALYSIS OF THE STOCHASTIC COLLOCATION CAMPAIGN

Execute once.
==============================================================================
"""

import easyvvuq as uq
import os
import fabsim3_cmd_api as fab
import matplotlib.pyplot as plt
import numpy as np

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
# work_dir = '/tmp'
config = 'PC_CI_SD_suppress_campaign2'

campaign = uq.Campaign(state_file="covid_easyvvuq_state.json", 
                       work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
# sampler.load_state("covid_sampler_state.pickle")
campaign.set_sampler(sampler)

#run the UQ ensemble
fab.get_uq_samples(config, campaign.campaign_dir, sampler._number_of_samples,
                   machine='eagle_vecma')
campaign.collate()

# Post-processing analysis
analysis = uq.analysis.SCAnalysis(
    sampler=campaign._active_sampler,
    qoi_cols=output_columns
)

campaign.apply_analysis(analysis)

results = campaign.get_last_analysis()

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="days", ylabel=output_columns[0])
mean = results["statistical_moments"]["cumDeath"]["mean"]
std = results["statistical_moments"]["cumDeath"]["std"]
ax.plot(mean)
ax.plot(mean + std, '--r')
ax.plot(mean - std, '--r')
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=[4, 4])
ax = fig.add_subplot(131, xlabel='days', ylabel=output_columns[0],
                      title='Surrogate samples')
ax.plot(analysis.get_sample_array(output_columns[0]).T, 'ro', alpha = 0.5)

#generate n_mc samples from the input distributions
n_mc = 20
xi_mc = np.zeros([n_mc,sampler.xi_d.shape[1]])
idx = 0
for dist in sampler.vary.get_values():
    xi_mc[:, idx] = dist.sample(n_mc)
    idx += 1
# xi_mc = sampler.xi_d
# n_mc = sampler.xi_d.shape[0]
    
# evaluate the surrogate at these values
print('Evaluating surrogate model', n_mc, 'times')
for i in range(n_mc):
    ax.plot(analysis.surrogate(output_columns[0], xi_mc[i]), 'g')
print('done')