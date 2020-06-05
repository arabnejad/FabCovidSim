# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabCovidsim.
#
# authors: Hamid Arabnejad, Wouter Edeling, Diana Suleimenova, Robert Sinclair

import easyvvuq as uq
import chaospy as cp
import os
import sys
import math
from pprint import pprint
import json
from shutil import copyfile, rmtree
import matplotlib.pyplot as plt


from plugins.FabCovidsim.FabCovidsim import *

# load custom Campaign
from plugins.FabCovidsim.customEasyVVUQ import CustomCampaign
uq.Campaign = CustomCampaign

# from customEasyVVUQ import CustomEncoder
# uq.encoders.GenericEncoder=CustomEncoder

work_dir_SC = os.path.join(os.path.dirname(__file__), 'covid_standard_test')
log_file = os.path.join(work_dir_SC, "log.txt")


'''
output_columns = ["R",
                  "SARI", "incSARI", "cumSARI",
                  "Critical", "incCritical", "cumCritical",
                  "incDeath", "cumDeath",
                  "incDeath_SARI", "incDeath_Critical",
                  "cumDeath_SARI", "cumDeath_Critical"]
'''
output_columns = ["cumDeath", "SARI", "incSARI", "cumSARI", "R", "Critical"]


@task
def covid_init_SC(config, ** args):
    '''
    fab eagle_vecma covid_init_SC:UK_easyvvuq_test

    '''
    # delete work_dir_SC is exists
    if os.path.exists(work_dir_SC):
        rmtree(work_dir_SC)
    os.mkdir(work_dir_SC)

    # Set up a fresh campaign called "covid-sim-standard"
    campaign = uq.Campaign(name='covid-sim-standard',
                           work_dir=work_dir_SC)

    # to make sure we are not overwriting the new simulation on previous ones
    job_label = campaign._campaign_dir

    # Define parameter space for the covid-sim-standard app
    params = json.load(open(get_plugin_path("FabCovidsim") +
                            '/templates/params_p_PC_CI_HQ_SD.json'))

    # params["mortality_factor"] = {"default": 1, "type": "float"}
    # params["p_symptomatic"] = {"default": 0.5, "type": "float"}

    # Create an encoder and decoder
    directory_tree = {'param_files': None}

    multiencoder_covid_sim = uq.encoders.MultiEncoder(
        uq.encoders.DirectoryBuilder(tree=directory_tree),
        uq.encoders.GenericEncoder(
            template_fname=get_plugin_path("FabCovidsim") +
            '/templates/template_p_PC_CI_HQ_SD.txt',
            delimiter='$',
            target_filename='param_files/p_PC_CI_HQ_SD.txt'
        ),
        uq.encoders.GenericEncoder(
            template_fname=get_plugin_path("FabCovidsim") +
            '/templates/template_preUK_R0=2.0.txt',
            delimiter='$',
            target_filename='param_files/preUK_R0=2.0.txt'
        )
    )

    decoder = uq.decoders.SimpleCSV(target_filename='output_dir/United_Kingdom_PC_CI_HQ_SD_R0=2.4.avNE.severity.xls',
                                    output_columns=output_columns,
                                    header=0,
                                    delimiter='\t')

    decoder2 = uq.decoders.SimpleCSV(target_filename='output_dir/United_Kingdom_NoInt_R0=2.4.avNE.severity.xls',
                                     output_columns=output_columns,
                                     header=0,
                                     delimiter='\t')
    collater = uq.collate.AggregateSamples(average=False)

    # parameters to vary
    vary = {
        "Relative_place_contact_rate_given_social_distancing_by_place_type0": cp.Uniform(0.8, 1.0),
        "Relative_place_contact_rate_given_social_distancing_by_place_type1": cp.Uniform(0.4, 0.6),
    }

    #=================================
    # create SCSampler
    #=================================
    sampler = uq.sampling.SCSampler(vary=vary,
                                    polynomial_order=1,
                                    quadrature_rule="G")

    # Add the app
    campaign.add_app(name="covidsim-standard-NoInt",
                     params=params,
                     encoder=multiencoder_covid_sim,
                     collater=collater,
                     decoder=decoder2)

    campaign.add_app(name="covidsim-standard-PC_CI_HQ_SD",
                     params=params,
                     encoder=multiencoder_covid_sim,
                     collater=collater,
                     decoder=decoder)

    # Set the active app to be covidsim-standard-PC_CI_HQ_SD
    campaign.set_app("covidsim-standard-NoInt")
    campaign.set_sampler(sampler)
    campaign.draw_samples()

    # Set the active app to be covidsim-standard-PC_CI_HQ_SD
    campaign.set_app("covidsim-standard-PC_CI_HQ_SD")
    campaign.set_sampler(sampler)
    campaign.draw_samples()

    campaign.set_app("covidsim-standard-NoInt")
    run_ids = campaign.populate_runs_dir()
    pprint(run_ids)

    campaign.set_app("covidsim-standard-PC_CI_HQ_SD")
    run_ids = campaign.populate_runs_dir()
    pprint(run_ids)

    # copy generated run folders to SWEEP directory in config folder
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/*" % (sweep_dir))
    local("cp -r %s/* %s" % (os.path.join(campaign.work_dir,
                                          campaign.campaign_dir,
                                          'SWEEP'
                                          ),
                             os.path.join(sweep_dir
                                          )
                             )
          )

    exit()
    # submit ensemble jobs to remote machine
    CovidSim_ensemble(config, label=job_label, **args)

    # save campaign state
    campaign.save_state(os.path.join(work_dir_SC, "campaign_state.json"))
    # sampler.save_state(os.path.join(work_dir_SC, "campaign_sampler.pickle"))


@task
def covid_analyse_SC(config, ** args):
    '''
    fab eagle_vecma covid_analyse_SC:UK_easyvvuq_test

    '''

    # reload Campaign
    campaign = uq.Campaign(state_file=os.path.join(work_dir_SC,
                                                   "campaign_state.json"),
                           work_dir=work_dir_SC
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')

    sampler = campaign.get_active_sampler()
    # sampler.load_state(os.path.join(work_dir_SC, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)

    # fetch only the required folder from remote machine
    with_config(config)
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))
    print("fetching results from remote machine ...")
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        fetch_results(regex=job_folder_name)
    print("Done\n")

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_SC, campaign._campaign_dir, 'SWEEP')

    print("Syncing output_dir ...")
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='/*/' \
            --include='/*/output_dir/***'  \
            --exclude='*' \
            {}/  {} ".format(src, des)
        )
    print("Done\n")

    campaign.collate()

    # Return dataframe containing all collated results
    collation_result = campaign.get_collation_result()

    collation_result.to_csv(os.path.join(work_dir_SC, 'collation_result.csv'),
                            index=False
                            )

    print(collation_result)

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(
        sampler=campaign._active_sampler,
        qoi_cols=output_columns
    )
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    # --------------------------------------------------------------------------
    #                   Plotting
    # --------------------------------------------------------------------------

    for output_column in output_columns:
        fig = plt.figure()
        ax = fig.add_subplot(111,
                             xlabel="days", ylabel=output_column)
        mean = results["statistical_moments"][output_column]["mean"]
        std = results["statistical_moments"][output_column]["std"]
        ax.plot(mean)
        ax.plot(mean + std, '--r')
        ax.plot(mean - std, '--r')
        #ax.title.set_text('statistical_moments for {}'.format(output_column))

        plt.tight_layout()
        plt.savefig(os.path.join(
            work_dir_SC, 'plot_statistical_moments_{}'.format(output_column)),
            dpi=400)

    '''
    # int conversions always round down
    nrows = int(math.sqrt(len(output_columns)))
    # If the number of plots is a perfect square, we're done.
    # Otherwise, we take the next highest perfect square to build our subplots
    if nrows**2 == len(output_columns):
        ncols = nrows
    else:
        #ncols = nrows + 1
        ncols = math.ceil(len(output_columns) / nrows)

    fig = plt.figure()

    plot_count = 1
    for output_column in output_columns:
        ax = fig.add_subplot(nrows, ncols, plot_count,
                             xlabel="days", ylabel=output_column)
        mean = results["statistical_moments"][output_column]["mean"]
        std = results["statistical_moments"][output_column]["std"]
        ax.plot(mean)
        ax.plot(mean + std, '--r')
        ax.plot(mean - std, '--r')
        #ax.title.set_text('statistical_moments for {}'.format(output_column))
        plot_count = plot_count + 1

    # plt.tight_layout()
    plt.savefig(os.path.join(work_dir_SC, 'plot_statistical_moments'), dpi=400)
    '''
