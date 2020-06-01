"""
==============================================================================
INITIALIZATION OF A (DIMENSION-ADAPTIVE) STOCHASTIC COLLOCATION CAMPAIGN

Execute once.
==============================================================================
"""

import easyvvuq as uq
import chaospy as cp
import os
import json
import fabsim3_cmd_api as fab
import numpy as np

from custom import CustomEncoder

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '~/postdoc1/covid/campaigns'
config = 'disease_adaptive'

# Set up a fresh campaign called "cannon"
campaign = uq.Campaign(name='covid', work_dir=work_dir)

# Define parameter space for the cannonsim app
params = json.load(open(home + '/../templates/params.json'))

params["Mortality_factor"] = {
                                "default": 1,
                                "type": "float"
                            }
params["Proportion_symptomatic"] = {
                                  "default": 0.5,
                                  "type": "float"
                              }
params["Relative_spatial_contact_rates_by_age_power"] = {
                                "default": 1,
                                "type": "float"
                            }


# Create an encoder and decoder
directory_tree = {'param_files': None}

multiencoder = uq.encoders.MultiEncoder(
    uq.encoders.DirectoryBuilder(tree=directory_tree),
    uq.encoders.GenericEncoder(
        template_fname=home + '/../templates/template_run_sample.py',
        delimiter = '$',
        target_filename='run_sample.py'),
    CustomEncoder(
        template_fname=home + '/../templates/template_p_PC_CI_HQ_SD.txt',
        target_filename='param_files/p_PC_CI_HQ_SD.txt'),
    CustomEncoder(
        template_fname=home + '/../templates/template_preGB_R0=2.0.txt',
        target_filename='param_files/preGB_R0=2.0.txt')
)

decoder = uq.decoders.SimpleCSV(
    target_filename='output_dir/United_Kingdom_PC_CI_HQ_SD.avNE.severity.xls', 
    output_columns=output_columns, header=0, delimiter='\t')

collater = uq.collate.AggregateSamples(average=False)

# Add the app
campaign.add_app(name="covid_robbie",
                 params=params,
                 encoder=multiencoder,
                 collater=collater,
                 decoder=decoder)
# Set the active app to be cannonsim (this is redundant when only one app
# has been added)
campaign.set_app("covid_robbie")

#parameters to vary
vary = {
        "Symptomatic_infectiousness_relative_to_asymptomatic": cp.Uniform(1,2.5),
        "Proportion_symptomatic": cp.Uniform(0.4,0.8),
        "Latent_period": cp.Uniform(3,7),
        "Mortality_factor": cp.Uniform(0.8,1.2),
        "Reproduction_number": cp.Uniform(2,3),
        "Infectious_period": cp.Uniform(11.5, 15.6),
        "Household_attack_rate": cp.Uniform(0.1, 0.19),
        "Household_transmission_denominator_power": cp.Uniform(0.7, 0.9),
        "Delay_from_end_of_latent_period_to_start_of_symptoms": cp.Uniform(0, 1.5),
        "Relative_transmission_rates_for_place_types0": cp.Uniform(0.08, 0.15),
        "Relative_transmission_rates_for_place_types1": cp.Uniform(0.08, 0.15),
        "Relative_transmission_rates_for_place_types2": cp.Uniform(0.05, 0.1),
        "Relative_transmission_rates_for_place_types3": cp.Uniform(0.05, 0.07),
        "Relative_spatial_contact_rates_by_age_power": cp.Uniform(0.25, 4)
}

#=================================
#create dimension-adaptive sampler
#=================================
#sparse = use a sparse grid (required)
#growth = use a nested quadrature rule (not required)
#midpoint_level1 = use a single collocation point in the 1st iteration (not required)
#dimension_adaptive = use a dimension adaptive sampler (required)
sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                quadrature_rule="C",
                                sparse=True, growth=True,
                                midpoint_level1=True,
                                dimension_adaptive=True)

campaign.set_sampler(sampler)
campaign.draw_samples()
campaign.populate_runs_dir()

campaign.save_state("covid_easyvvuq_state.json")
sampler.save_state("covid_sampler_state.pickle")

#run the UQ ensemble
fab.run_uq_ensemble(config, campaign.campaign_dir, script='CovidSim',
                    machine="eagle_vecma", PilotJob = False)
