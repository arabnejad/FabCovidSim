"""
==============================================================================

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
config = 'PC_CI_HQ_SD_suppress_campaign2'

# Set up a fresh campaign called "cannon"
campaign = uq.Campaign(name='covid', work_dir=work_dir)

# Define parameter space for the cannonsim app
params = json.load(open(home + '/../templates_campaign2/params.json'))

directory_tree = {'param_files': None}
multiencoder_p_PC_CI_HQ_SD = uq.encoders.MultiEncoder(
    uq.encoders.DirectoryBuilder(tree=directory_tree),
    uq.encoders.GenericEncoder(         
        template_fname=home + '/../templates_campaign2/p_PC_CI_HQ_SD.txt',
        delimiter='$',
        target_filename='param_files/p_PC_CI_HQ_SD.txt'),
    uq.encoders.GenericEncoder(
        template_fname=home + '/../templates_campaign2/preGB_R0=2.0.txt',
        delimiter='$',
        target_filename='param_files/preGB_R0=2.0.txt'),
    uq.encoders.GenericEncoder(
        template_fname=home + '/../templates_campaign2/p_seeds.txt',
        delimiter='$',
        target_filename='param_files/p_seeds.txt')
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
vary = {'Random_seeds2':cp.DiscreteUniform(100000, 150000),
        'Random_seeds3':cp.DiscreteUniform(150001, 200000)}

#=================================
#create dimension-adaptive sampler
#=================================
#sparse = use a sparse grid (required)
#growth = use a nested quadrature rule (not required)
#midpoint_level1 = use a single collocation point in the 1st iteration (not required)
#dimension_adaptive = use a dimension adaptive sampler (required)
sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=5,
                                quadrature_rule="C",
                                sparse=False, growth=False)
campaign.set_sampler(sampler)

print('Number of samples = %d' % sampler._number_of_samples)

campaign.draw_samples()
campaign.populate_runs_dir()

campaign.save_state("covid_easyvvuq_state.json")
sampler.save_state("covid_sampler_state.pickle")

# run the UQ ensemble
fab.run_uq_ensemble(config, campaign.campaign_dir, script='CovidSim',
                    machine="eagle_vecma", PilotJob = False)