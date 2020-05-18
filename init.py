import easyvvuq as uq
import chaospy as cp
import os
import json
import fabsim3_cmd_api as fab

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
config = 'UK_easyvvuq_test'

# Set up a fresh campaign called "cannon"
campaign = uq.Campaign(name='covid', work_dir=work_dir)

# Define parameter space for the cannonsim app
params_p_PC7_CI_HQ_SD = json.load(open(home + '/templates/params_p_PC7_CI_HQ_SD.json'))

# Create an encoder and decoder
directory_tree = {'param_files': None}

multiencoder_p_PC7_CI_HQ_SD = uq.encoders.MultiEncoder(
    uq.encoders.DirectoryBuilder(tree=directory_tree),
    uq.encoders.GenericEncoder(
        template_fname=home + '/templates/template_p_PC7_CI_HQ_SD.txt',
        delimiter='$',
        target_filename='param_files/template_p_PC7_CI_HQ_SD.txt'),
    uq.encoders.GenericEncoder(
        template_fname=home + '/templates/template_preUK_R0=2.0.txt',
        delimiter='$',
        target_filename='param_files/preUK_R0=2.0.txt')
)

decoder = uq.decoders.SimpleCSV(
    target_filename='???', output_columns=output_columns, header=0)

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

# Create a collation element for this campaign

vary = {
    "Household_level_compliance_with_quarantine": cp.Uniform(0.3, 0.75),
    "Symptomatic_infectiousness_relative_to_asymptomatic": cp.Uniform(1.3, 1.70),
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

campaign.save_state(os.path.join(
    campaign.campaign_dir, "covid_easyvvuq_state.json"))
sampler.save_state(os.path.join(
    campaign.campaign_dir, "covid_sampler_state.pickle"))

fab.run_uq_ensemble(config, campaign.campaign_dir, machine="eagle_vecma")