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
import pytest
import math
from pprint import pprint
import subprocess
import json
from shutil import copyfile, rmtree
from sqlalchemy import create_engine
import re
import pandas as pd
import matplotlib.pyplot as plt

try:
    import glob
except ImportError:
    raise ImportError('python glob module NOT found !! ')


from plugins.FabCovidsim.FabCovidsim import *

# load custom Campaign
from plugins.FabCovidsim.customEasyVVUQ import CustomCampaign, CustomSCAnalysis
uq.Campaign = CustomCampaign
uq.analysis.SCAnalysis = CustomSCAnalysis

'''
order to be executed
    fab eagle_vecma covid_init:GB_suppress
    fab eagle_vecma covid_analyse:GB_suppress
    loop
        fab eagle_vecma covid_look_ahead:GB_suppress
        fab eagle_vecma covid_adapt:GB_suppress

'''
work_dir_adapt = os.path.join(os.path.dirname(__file__), 'covid_adaptive_test')
log_file = os.path.join(work_dir_adapt, "log.txt")
backup_dir = os.path.join(work_dir_adapt, 'backup')

'''
output_columns = ["R",
                  "SARI", "incSARI", "cumSARI",
                  "Critical", "incCritical", "cumCritical",
                  "incDeath", "cumDeath",
                  "incDeath_SARI", "incDeath_Critical",
                  "cumDeath_SARI", "cumDeath_Critical"]
'''
output_columns = ["cumDeath"]


class custom_redirection(object):

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f in self.files:
            f.flush()


@task
def covid_adapt(config, ** args):
    '''
    fab eagle_vecma covid_adapt:GB_suppress
    '''
    load_campaign_files()

    # reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file=os.path.join(work_dir_adapt,
                                                   "campaign_state.json"),
                           work_dir=work_dir_adapt
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')
    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # fetch only the required folder from remote machine
    with_config(config)
    # env.job_name_template += "_{}".format(job_label)
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))

    print("fetching results from remote machine ...")
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        fetch_results(regex=job_folder_name)
    print("Done\n")

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')
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

    # compute the error at all admissible points, select direction with
    # highest error and add that direction to the grid
    data_frame = campaign.get_collation_result()
    # for output_column in output_columns[0]:
    for output_column in [output_columns[0]]:
        analysis.adapt_dimension(output_column, data_frame)

    # save everything
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    analysis.save_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # apply analysis
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    # for output_column in output_columns:
    for output_column in [output_columns[0]]:
        #########################
        # plot mean +/- std dev #
        #########################
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel="days", ylabel=output_column)
        mean = results["statistical_moments"][output_column]["mean"]
        std = results["statistical_moments"][output_column]["std"]
        ax.plot(mean)
        ax.plot(mean + std, '--r')
        ax.plot(mean - std, '--r')
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir_adapt,
                                 'plot_mean_std_%d[%s]' %
                                 (sampler.number_of_adaptations,
                                  output_column)
                                 ),
                    dpi=400)

        #################################
        # Plot some convergence metrics #
        #################################
        # plot max quad order per dimension. Gives an idea of which
        # variables are important
        analysis.adaptation_histogram(
            os.path.join(work_dir_adapt,
                         'plot_adaptation_histogram_%d[%s]'
                         % (sampler.number_of_adaptations, output_column)
                         )
        )

        analysis.plot_stat_convergence(
            os.path.join(work_dir_adapt,
                         'plot_stat_convergence%d[%s]'
                         % (sampler.number_of_adaptations, output_column)
                         )
        )

        surplus_errors = analysis.get_adaptation_errors()

        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='refinement step',
                             ylabel='max surplus error')
        ax.plot(range(1, len(surplus_errors) + 1), surplus_errors, '-b*')
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir_adapt,
                                 'max_surplus_error_%d[%s]' %
                                 (sampler.number_of_adaptations,
                                  output_column)
                                 ),
                    dpi=400)

        #####################################
        # Plot the random surrogate samples #
        #####################################

        fig = plt.figure(figsize=[12, 4])
        ax = fig.add_subplot(131, xlabel='days', ylabel=output_column,
                             title='Surrogate samples')
        ax.plot(analysis.get_sample_array(
            output_column).T, 'ro', alpha=0.5)

        # generate n_mc samples from the input distributions
        n_mc = 20
        xi_mc = np.zeros([n_mc, sampler.xi_d.shape[1]])
        idx = 0
        for dist in sampler.vary.get_values():
            xi_mc[:, idx] = dist.sample(n_mc)
            idx += 1
        xi_mc = sampler.xi_d
        n_mc = sampler.xi_d.shape[0]

        # evaluate the surrogate at these values
        print('Evaluating surrogate model', n_mc, 'times')
        for i in range(n_mc):
            ax.plot(analysis.surrogate(output_column, xi_mc[i]), 'g')
        print('done')

        plt.savefig(os.path.join(work_dir_adapt,
                                 'Surrogate_samples_%d[%s]' %
                                 (sampler.number_of_adaptations,
                                  output_column)
                                 ),
                    dpi=400)

        ##################################
        # Plot first-order Sobol indices #
        ##################################

        ax = fig.add_subplot(122, title=r'First-order Sobols indices',
                             xlabel="days", ylabel=output_column)
        sobols_first = results["sobols_first"][output_column]
        for param in sobols_first.keys():
            ax.plot(sobols_first[param], label=param)
        leg = ax.legend(loc=0, fontsize=8)
        leg.set_draggable(True)
        plt.tight_layout()

        plt.savefig(os.path.join(work_dir_adapt, 'plot_first_order_Sobol_indices_%d' %
                                 (sampler.number_of_adaptations)), dpi=400)

    backup_campaign_files()


@task
def covid_look_ahead(config, ** args):
    '''
    fab eagle_vecma covid_look_ahead:GB_suppress
    '''

    load_campaign_files()

    # reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file=os.path.join(work_dir_adapt,
                                                   "campaign_state.json"),
                           work_dir=work_dir_adapt
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')
    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # look-ahead step (compute the code at admissible forward points)
    sampler.look_ahead(analysis.l_norm)

    # proceed as usual
    campaign.draw_samples()
    run_ids = campaign.populate_runs_dir()

    # copy generated run folders to SWEEP directory in config folder
    # clean SWEPP, this part should be added to FabSim3.campaign2ensemble
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/*" % (sweep_dir))
    # campaign2ensemble copies all run_ids, we don't need it here,
    # only new run_id generated at this step should be copied
    for run_id in run_ids:
        local("cp -r %s %s" % (os.path.join(campaign.work_dir,
                                            campaign.campaign_dir,
                                            'SWEEP',
                                            run_id
                                            ),
                               os.path.join(sweep_dir,
                                            run_id
                                            )
                               )
              )

    # submit ensemble jobs to remote machine
    # run the UQ ensemble at the admissible forward points
    job_label = campaign._campaign_dir
    CovidSim_ensemble(config, label=job_label, **args)

    # save campaign and sampler
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))

    backup_campaign_files()


@task
def covid_analyse(config, ** args):
    '''
    fab eagle_vecma covid_analyse:GB_suppress
    '''

    load_campaign_files()

    # reload Campaign, sampler, analysis
    campaign = uq.Campaign(state_file=os.path.join(work_dir_adapt,
                                                   "campaign_state.json"),
                           work_dir=work_dir_adapt
                           )
    print('========================================================')
    print('Reloaded campaign', campaign._campaign_dir)
    print('========================================================')

    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    campaign.set_sampler(sampler)

    # fetch only the required folder from remote machine
    with_config(config)
    # env.job_name_template += "_{}".format(job_label)
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))
    fetch_results(regex=job_folder_name)

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')
    local(
        "rsync -av -m -v \
        --include='/*/' \
        --include='/*/output_dir/***'  \
        --exclude='*' \
        {}/  {} ".format(src, des)
    )

    campaign.collate()

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(
        sampler=campaign._active_sampler,
        qoi_cols=output_columns
    )

    campaign.apply_analysis(analysis)

    # save analysis state
    # this is a temporary subroutine which saves the entire state of
    # the analysis in a pickle file. The proper place for this is the database
    analysis.save_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    backup_campaign_files()


@task
def covid_init(config, ** args):
    '''
    fab eagle_vecma covid_init:GB_suppress
    '''

    # delete work_dir is exists
    if os.path.exists(work_dir_adapt):
        rmtree(work_dir_adapt)
    os.mkdir(work_dir_adapt)

    # Set up a fresh campaign called "covid-sim-adaptive"
    campaign = uq.Campaign(name='covid-sim-adaptive',
                           work_dir=work_dir_adapt)

    # to make sure we are not overwriting the new simulation on previous ones
    job_label = campaign._campaign_dir

    # Define parameter space for the covid-sim-adaptive app
    params = json.load(open(os.path.join(get_plugin_path("FabCovidsim"),
                                         'templates',
                                         config,
                                         'params.json'
                                         )
                            )
                       )

    # Create an encoder and decoder
    directory_tree = {'param_files': None}

    multiencoder_covid_sim = uq.encoders.MultiEncoder(
        uq.encoders.DirectoryBuilder(tree=directory_tree),
        uq.encoders.GenericEncoder(
            template_fname=os.path.join(get_plugin_path("FabCovidsim"),
                                        'templates',
                                        config,
                                        'template_p_PC_CI_HQ_SD.txt'
                                        ),
            delimiter='$',
            target_filename='param_files/p_PC_CI_HQ_SD.txt'
        ),
        uq.encoders.GenericEncoder(
            template_fname=os.path.join(get_plugin_path("FabCovidsim"),
                                        'templates',
                                        config,
                                        'template_preGB_R0=2.0.txt'),
            delimiter='$',
            target_filename='param_files/preGB_R0=2.0.txt'
        ),
        uq.encoders.GenericEncoder(
            template_fname=os.path.join(get_plugin_path("FabCovidsim"),
                                        'templates',
                                        config,
                                        'template_run_sample.py'),
            delimiter='$',
            target_filename='run_sample.py'
        )
    )

    decoder = uq.decoders.SimpleCSV(target_filename='output_dir/United_Kingdom_PC_CI_HQ_SD_R0=2.4.avNE.severity.xls',
                                    output_columns=output_columns,
                                    header=0,
                                    delimiter='\t')

    collater = uq.collate.AggregateSamples(average=False)

    '''
    [Duration of place closure]
    #2
    [Trigger incidence per cell for place closure]
    #1
    --------------
    If 
    [Proportion compliant with enhanced social distancing] = 0 , 
    all parameters below are not considered, which is currently the case.
    --------------
    [Trigger incidence per cell for end of place closure]
    #5
    [Duration of household quarantine policy]
    #4
    [Duration of case isolation policy]
    #4
    [Trigger incidence per cell for social distancing]
    #1
    [Trigger incidence per cell for end of social distancing]
    #5
    [Duration of social distancing]
    #2
    '''

    params["Proportion_compliant_with_enhanced_social_distancing"]["default"] = 1
    params["R0"]["default"] = 2.4

    '''
    Off_trigger_as_proportion_of_on_trigger="60 100 200 300 400"
    On_trigger="0.25 0.5 0.75"
    '''

    # Add the app
    campaign.add_app(name="covidsim-adaptive-PC_CI_HQ_SD",
                     params=params,
                     encoder=multiencoder_covid_sim,
                     collater=collater,
                     decoder=decoder)

    # Set the active app to be covidsim-adaptive-PC_CI_HQ_SD
    campaign.set_app("covidsim-adaptive-PC_CI_HQ_SD")

    # parameters to vary
    vary = {
        "Relative_place_contact_rate_given_social_distancing_by_place_type0": cp.Uniform(0.8, 1.0),
        "Relative_place_contact_rate_given_social_distancing_by_place_type1": cp.Uniform(0.6, 0.9),
        "Proportion_of_places_remaining_open_after_closure_by_place_type2": cp.Uniform(0.2, 0.3),
        "Proportion_of_places_remaining_open_after_closure_by_place_type3": cp.Uniform(0.9, 0.1),
        "Residual_place_contacts_after_household_quarantine_by_place_type0": cp.Uniform(0.2, 0.3),
        "Relative_household_contact_rate_given_enhanced_social_distancing": cp.Uniform(1.0, 1.4),
        "Relative_place_contact_rate_given_enhanced_social_distancing_by_place_type0": cp.Uniform(0.15, 0.5),
        "Relative_place_contact_rate_given_enhanced_social_distancing_by_place_type1": cp.Uniform(0.15, 0.5),
        "Relative_place_contact_rate_given_enhanced_social_distancing_by_place_type2": cp.Uniform(0.15, 0.5),
        "Relative_place_contact_rate_given_enhanced_social_distancing_by_place_type3": cp.Uniform(0.15, 0.5),
        "Off_trigger_as_proportion_of_on_trigger": cp.Uniform(50, 450),
        "On_trigger": cp.Uniform(0.2, 0.8),
    }

    #=================================
    # create dimension-adaptive sampler
    #=================================
    # sparse = use a sparse grid (required)
    # growth = use a nested quadrature rule (not required)
    # midpoint_level1 = use a single collocation point in the 1st iteration (not required)
    # dimension_adaptive = use a dimension adaptive sampler (required)
    sampler = uq.sampling.SCSampler(vary=vary,
                                    polynomial_order=1,
                                    quadrature_rule="C",
                                    sparse=True,
                                    growth=True,
                                    midpoint_level1=True,
                                    dimension_adaptive=True
                                    )

    campaign.set_sampler(sampler)
    print('Number of samples = %d' % (sampler._number_of_samples),
          file=open(log_file, 'a'))
    campaign.draw_samples()
    run_ids = campaign.populate_runs_dir()

    # copy generated run folders to SWEEP directory in config folder
    # clean SWEPP, this part should be added to FabSim3.campaign2ensemble
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/*" % (sweep_dir))
    # campaign2ensemble copies all run_ids, we don't need it here,
    # only new run_id generated at this step should be copied
    for run_id in run_ids:
        local("cp -r %s %s" % (os.path.join(campaign.work_dir,
                                            campaign.campaign_dir,
                                            'SWEEP',
                                            run_id
                                            ),
                               os.path.join(sweep_dir,
                                            run_id
                                            )
                               )
              )

    # submit ensemble jobs to remote machine
    CovidSim_ensemble(config, label=job_label, **args)

    # save campaign and sampler state
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    backup_campaign_files()


def backup_campaign_files():

    # delete backup folder
    if os.path.exists(backup_dir):
        rmtree(backup_dir)
    os.mkdir(backup_dir)
    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(work_dir_adapt, backup_dir)
        )


def load_campaign_files():

    with hide('output', 'running', 'warnings'), settings(warn_only=True):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(backup_dir, work_dir_adapt)
        )
