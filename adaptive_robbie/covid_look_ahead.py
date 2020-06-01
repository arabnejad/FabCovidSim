"""
==============================================================================
THE LOOK-AHEAD STEP

Takes the current configuration of points, computes so-called adimissble
points in the look_forward subroutine. These points are used in 
the adaptation step to decide along which dimension to place more samples.

The look-ahead step and the adaptation step can be executed multiple times:
look ahead, adapt, look ahead, adapt, etc
==============================================================================
"""

import easyvvuq as uq
import os
import fabsim3_cmd_api as fab

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '~/postdoc1/covid/campaigns'
config = 'disease_adaptive'

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

#required parameter in the case of a Fabsim run
skip = sampler.count

#look-ahead step (compute the code at admissible forward points)
sampler.look_ahead(analysis.l_norm)

#proceed as usual
campaign.draw_samples()
campaign.populate_runs_dir()

#save campaign and sampler
campaign.save_state("covid_easyvvuq_state.json")
sampler.save_state("covid_sampler_state.pickle")

#run the UQ ensemble at the admissible forward points
#skip (int) = the number of previous samples: required to avoid recomputing
#already computed samples from a previous iteration
fab.run_uq_ensemble(config, campaign.campaign_dir, script='CovidSim',
                    machine="eagle_vecma", skip=skip, PilotJob=True)
