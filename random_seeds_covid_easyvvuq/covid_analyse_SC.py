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

#########################
# Plot mean +/- Std Dev #
#########################

fig = plt.figure()
ax = fig.add_subplot(111, xlabel="days", ylabel=output_columns[0])
mean = results["statistical_moments"]["cumDeath"]["mean"]
std = results["statistical_moments"]["cumDeath"]["std"]
ax.plot(mean)
ax.plot(mean + std, '--r')
ax.plot(mean - std, '--r')
plt.tight_layout()
plt.show()


#############################
# Plot confidence intervals #
#############################

from matplotlib import gridspec
import seaborn as sns

x = range(analysis.N_qoi)

surr_samples = analysis.get_sample_array(output_columns[0])
n_samples = surr_samples.shape[0]

#confidence bounds
lower1, upper1 = analysis.get_confidence_intervals(output_columns[0], n_samples, conf=0.63,
                                                    surr_samples=surr_samples)
lower2, upper2 = analysis.get_confidence_intervals(output_columns[0], n_samples, conf=0.95,
                                                    surr_samples=surr_samples)

fig = plt.figure(figsize=(10,5))
spec = gridspec.GridSpec(ncols=2, nrows=1,
                          width_ratios=[3, 1])

# ax1 = fig.add_subplot(spec[0], xlim=[0, 840], ylim=[-100, 80000])
ax1 = fig.add_subplot(spec[0])
ax2 = fig.add_subplot(spec[1], sharey=ax1)
ax2.get_xaxis().set_ticks([])
fig.subplots_adjust(wspace=0)
plt.setp(ax2.get_yticklabels(), visible=False)

ax1.fill_between(x, lower2, upper2, color='#aa99cc', label='95% CI', alpha=0.5)
ax1.fill_between(x, lower1, upper1, color='#aa99cc', label='68% CI')

mean = results["statistical_moments"][output_columns[0]]["mean"]
ax1.plot(x, mean, label='Mean')

# #plot a single sample of report 9
# ax1.plot(cumDeath_rep9, '--', color='#ffb20a', label=r'Baseline report 9 sample', linewidth=3)

# #plot data
# ax1.plot(np.arange(day_start, day_start + cumDeath_data.size)[0:-1:7],
#          cumDeath_data[0:-1:7], 's', color='olivedrab', label='Data')

ax1.legend(loc="upper left")

ax1.set_xlabel('Days')
ax1.set_ylabel('Cumulative deaths')
# ax2.set_xlabel('Frequency')
#ax2.set_title('Total deaths distribution')
ax2.axis('off')

total_deaths = surr_samples[:, -1]
ax2 = sns.distplot(total_deaths, vertical=True)

plt.tight_layout()
##################################
# Plot first-order Sobol indices #
##################################

fig = plt.figure(figsize=(7,5))
spec = gridspec.GridSpec(ncols=2, nrows=1,
                          width_ratios=[3, 1])

# ax1 = fig.add_subplot(spec[0], xlim=[0, 840], ylim=[-100, 80000])
ax = fig.add_subplot(spec[0], title=r'First-order Sobol indices',
                      xlabel="days", ylabel=r'$S_i$', ylim=[0,1])

from itertools import cycle

sobols, D_u = analysis.get_sobol_indices(output_columns[0], typ='all')

# color = cycle(['b', 'r', 'g', 'm', 'c', 'k'])
marker = cycle(['o', 'v', '^', '<', '>', 's', '*', 'p', 'd', 'P', 'X'])
skip = 30
x = range(0, analysis.N_qoi, skip)

sobols_first = results["sobols_first"][output_columns[0]]

first_order_contribution = 0
for param in sobols_first.keys():
    first_order_contribution += sobols_first[param][0:-1:skip]

total_contribution = 0
for param in sobols.keys():
    ax.plot(x, sobols[param][0:-1:skip], label=param, marker=next(marker))
    total_contribution += sobols[param][0:-1:skip]
    
ax.plot(x, first_order_contribution, 'b*', label=r'First-order contribution')
ax.plot(x, total_contribution, 'rs', label=r'Total contribution')

leg = ax.legend(loc=0, fontsize=8)
leg.set_draggable(True)
plt.tight_layout()

