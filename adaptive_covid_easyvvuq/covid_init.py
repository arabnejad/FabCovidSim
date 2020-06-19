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

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
config = 'PC_CI_HQ_SD_suppress_campaign3_1'

# Set up a fresh campaign
campaign = uq.Campaign(name='covid', work_dir=work_dir)

# Define parameter space for the cannonsim app
params = json.load(open(home + '/../templates_campaign3_1/params.json'))

#for this campaign modify the random seeds manually, once
params['Random_seeds0']['default'] = 98798250
params['Random_seeds1']['default'] = 729201
params['Random_seeds2']['default'] = 17389301
params['Random_seeds3']['default'] = 4797332

# Create an encoder and decoder
directory_tree = {'param_files': None}

multiencoder_p_PC_CI_HQ_SD = uq.encoders.MultiEncoder(
    uq.encoders.DirectoryBuilder(tree=directory_tree),
    uq.encoders.GenericEncoder(         
        template_fname=home + '/../templates_campaign3_1/p_PC_CI_HQ_SD.txt',
        delimiter='$',
        target_filename='param_files/p_PC_CI_HQ_SD.txt'),
    uq.encoders.GenericEncoder(
        template_fname=home + '/../templates_campaign3_1/preGB_R0=2.0.txt',
        delimiter='$',
        target_filename='param_files/preGB_R0=2.0.txt')
)

decoder = uq.decoders.SimpleCSV(
    target_filename='output_dir/United_Kingdom_PC_CI_HQ_SD_R0=2.4.avNE.severity.xls', 
    output_columns=output_columns, header=0, delimiter='\t')

collater = uq.collate.AggregateSamples(average=False)

# Add the app
campaign.add_app(name=config,
                 params=params,
                 encoder=multiencoder_p_PC_CI_HQ_SD,
                 collater=collater,
                 decoder=decoder)
# Set the active app 
campaign.set_app(config)

#parameters to vary
vary = {
    # "Proportion_of_places_remaining_open_after_closure_by_place_type2": cp.Uniform(0.2, 0.3),
    # "Proportion_of_places_remaining_open_after_closure_by_place_type3": cp.Uniform(0.9, 1.0),
    "Relative_household_contact_rate_after_closure": cp.Uniform(1.5*0.8, 1.5*1.2),
    "Relative_spatial_contact_rate_after_closure": cp.Uniform(1.25*0.8, 1.25*1.2),
    "Relative_household_contact_rate_after_quarantine": cp.Uniform(2.0*0.8, 2.0*1.2),
    # "Residual_place_contacts_after_household_quarantine_by_place_type0": cp.Uniform(0.8*0.25, 1.2*0.25),
    # "Residual_place_contacts_after_household_quarantine_by_place_type1": cp.Uniform(0.8*0.25, 1.2*0.25),
    # "Residual_place_contacts_after_household_quarantine_by_place_type2": cp.Uniform(0.8*0.25, 1.2*0.25),
    # "Residual_place_contacts_after_household_quarantine_by_place_type3": cp.Uniform(0.8*0.25, 1.2*0.25),
    "Residual_spatial_contacts_after_household_quarantine": cp.Uniform(0.25*0.8, 0.25*1.2),
    "Household_level_compliance_with_quarantine": cp.Uniform(0.5, 0.9),
    "Individual_level_compliance_with_quarantine": cp.Uniform(0.9, 1.0),
    "Proportion_of_detected_cases_isolated":cp.Uniform(0.6, 0.8),
    "Residual_contacts_after_case_isolation":cp.Uniform(0.25*0.8, 0.25*1.2),
    "Relative_household_contact_rate_given_social_distancing":cp.Uniform(1.1, 1.25*1.2),
    "Relative_spatial_contact_rate_given_social_distancing":cp.Uniform(0.15, 0.35),
    "Delay_to_start_household_quarantine":cp.DiscreteUniform(1, 3),
    "Length_of_time_households_are_quarantined":cp.DiscreteUniform(12, 16),
    "Delay_to_start_case_isolation":cp.DiscreteUniform(1, 3),
    "Duration_of_case_isolation":cp.DiscreteUniform(5, 9)    
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

print('Number of samples = %d' % sampler._number_of_samples)

campaign.draw_samples()
campaign.populate_runs_dir()

campaign.save_state("covid_easyvvuq_state.json")
sampler.save_state("covid_sampler_state.pickle")

# run the UQ ensemble
fab.run_uq_ensemble(config, campaign.campaign_dir, script='CovidSim',
                    machine="eagle_vecma", PilotJob = False)
