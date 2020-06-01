"""
==============================================================================
ANALYSIS OF THE INITIAL (DIMENSION-ADAPTIVE) STCOHASTIC COLLOCATION CAMPAIGN

Execute once.
==============================================================================
"""

import easyvvuq as uq
import os
import fabsim3_cmd_api as fab

from custom import CustomEncoder

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '~/postdoc1/covid/campaigns'
config = 'disease_adaptive'

campaign = uq.Campaign(state_file="covid_easyvvuq_state.json", 
                       work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
sampler.load_state("covid_sampler_state.pickle")
campaign.set_sampler(sampler)

#run the UQ ensemble
fab.get_uq_samples(config, campaign.campaign_dir, sampler._number_of_samples,
                   machine='eagle_vecma')
print('here')
campaign.collate()
print('collaged')
print(campaign)

# Post-processing analysis
analysis = uq.analysis.SCAnalysis(
    sampler=campaign._active_sampler,
    qoi_cols=output_columns
)

campaign.apply_analysis(analysis)

#this is a temporary subroutine which saves the entire state of
#the analysis in a pickle file. The proper place for this is the database
analysis.save_state("covid_analysis_state.pickle")
