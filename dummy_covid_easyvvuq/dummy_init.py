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
work_dir = '/tmp'
config = 'dummy_covid'

# Set up a fresh campaign called "cannon"
campaign = uq.Campaign(name='covid', work_dir=work_dir)

# Define parameter space 
params_p_PC7_CI_HQ_SD = json.load(open(home + '/../templates/params_p_PC7_CI_HQ_SD.json'))

# Create an encoder and decoder
directory_tree = {'param_files': None}

multiencoder_p_PC7_CI_HQ_SD = uq.encoders.MultiEncoder(
    uq.encoders.DirectoryBuilder(tree=directory_tree),
    uq.encoders.GenericEncoder(
        template_fname=home + '/../templates/template_p_PC7_CI_HQ_SD.txt',
        delimiter='$',
        target_filename='param_files/template_p_PC7_CI_HQ_SD.txt'),
    uq.encoders.GenericEncoder(
        template_fname=home + '/../templates/template_preUK_R0=2.0.txt',
        delimiter='$',
        target_filename='param_files/preUK_R0=2.0.txt')
)

decoder = uq.decoders.SimpleCSV(
    target_filename='output_dir/foo.severity.xls', 
    output_columns=output_columns, header=0, delimiter='\t')

collater = uq.collate.AggregateSamples(average=False)

# Add the app
campaign.add_app(name="covid_p_PC7_CI_HQ_SD",
                 params=params_p_PC7_CI_HQ_SD,
                 encoder=multiencoder_p_PC7_CI_HQ_SD,
                 collater=collater,
                 decoder=decoder)
# Set the active app to be cannonsim (this is redundant when only one app
# has been added)
campaign.set_app("covid_p_PC7_CI_HQ_SD")

#parameters to vary
vary = {
    "Relative_household_contact_rate_after_closure": cp.Uniform(1.5*0.8, 1.5*1.2),
    "Relative_spatial_contact_rate_after_closure": cp.Uniform(1.25*0.8, 1.25*1.2),
    "Relative_household_contact_rate_after_quarantine": cp.Uniform(1.5*0.8, 1.5*1.2),
    "Residual_spatial_contacts_after_household_quarantine": cp.Uniform(0.25*0.8, 0.25*1.2),
    "Household_level_compliance_with_quarantine": cp.Uniform(0.5, 0.9),
    "Individual_level_compliance_with_quarantine": cp.Uniform(0.9, 1.0),
    "Proportion_of_detected_cases_isolated":cp.Uniform(0.85, 0.95),
    "Residual_contacts_after_case_isolation":cp.Uniform(0.25*0.8, 0.25*1.2),
    "Relative_household_contact_rate_given_social_distancing":cp.Uniform(1.1, 1.25*1.2),
    "Relative_spatial_contact_rate_given_social_distancing":cp.Uniform(0.05, 0.15)
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

if not os.path.exists('states'):
    os.makedirs('states')
campaign.save_state("states/covid_easyvvuq_state.json")
sampler.save_state("states/covid_sampler_state.pickle")

#run the UQ ensemble
fab.run_uq_ensemble(config, campaign.campaign_dir, script='Dummy_CovidSim',
                    machine="localhost")