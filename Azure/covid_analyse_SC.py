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

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
config = 'UK_easyvvuq_test'

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