# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabCovidsim.
#
# authors: Hamid Arabnejad, Diana Suleimenova, Robert Sinclair, Wouter Edeling.

import easyvvuq as uq
import chaospy as cp
import os
import sys
import pytest
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

tmp_dir = '/tmp/'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
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


# @task
def init_covid_campaign():

    # Set up a fresh campaign called "cannon"
    my_campaign = uq.Campaign(name='covid', work_dir=tmp_dir)

    # Define parameter space for the cannonsim app
    params_p_PC7_CI_HQ_SD = json.load(open(get_plugin_path(
        "FabCovidsim") + '/templates/params_p_PC7_CI_HQ_SD.json'))

    # Create an encoder and decoder
    directory_tree = {'param_files': None}

    multiencoder_p_PC7_CI_HQ_SD = uq.encoders.MultiEncoder(
        uq.encoders.DirectoryBuilder(tree=directory_tree),
        uq.encoders.GenericEncoder(
            template_fname=get_plugin_path(
                "FabCovidsim") + '/templates/template_p_PC7_CI_HQ_SD.txt',
            delimiter='$',
            target_filename='param_files/template_p_PC7_CI_HQ_SD.txt'),
        uq.encoders.GenericEncoder(
            template_fname=get_plugin_path(
                "FabCovidsim") + '/templates/template_preUK_R0=2.0.txt',
            delimiter='$',
            target_filename='param_files/preUK_R0=2.0.txt')
    )

    decoder = uq.decoders.SimpleCSV(
        target_filename='???', output_columns=output_columns, header=0)

    collater = uq.collate.AggregateSamples(average=False)

    # Add the app
    my_campaign.add_app(name="covid_p_PC7_CI_HQ_SD",
                        params=params_p_PC7_CI_HQ_SD,
                        encoder=multiencoder_p_PC7_CI_HQ_SD,
                        collater=collater,
                        decoder=decoder)
    # Set the active app to be cannonsim (this is redundant when only one app
    # has been added)
    my_campaign.set_app("covid_p_PC7_CI_HQ_SD")

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
    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                       quadrature_rule="C",
                                       sparse=True, growth=True,
                                       midpoint_level1=True,
                                       dimension_adaptive=True)

    my_campaign.set_sampler(my_sampler)
    my_campaign.draw_samples()
    my_campaign.populate_runs_dir()

    return my_campaign, my_sampler

@task
def look_ahead_covid_campaign(config, ** args):
    # make sure you run fetch_results() command before this using this function
    with_config(config)
    work_dir = os.path.join(env.local_results, template(env.job_name_template))
    json_file = rearrange_files(work_dir)
    #load campaign
    covidsim_campaign = uq.Campaign(state_file=json_file, work_dir=work_dir)    

    print('========================================================')
    print('Reloaded campaign', covidsim_campaign.campaign_dir.split('/')[-1])
    print('========================================================')

    covidsim_campaign.collate()    

    sampler = covidsim_campaign.get_active_sampler()
    sampler.load_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_sampler_state.pickle"))
    covidsim_campaign.set_sampler(sampler)

    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_analysis_state.pickle"))

    #look-ahead step (compute the code at admissible forward points)
    sampler.look_ahead(analysis.l_norm)

    #proceed as usual
    covidsim_campaign.draw_samples()
    covidsim_campaign.populate_runs_dir()

    covidsim_campaign.save_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_state.json"))

    #this is a temporary subroutine which saves the entire state of
    #the sampler in a pickle file. The proper place for this is the database
    sampler.save_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_sampler_state.pickle"))

@task
def adapt_covid_campaign(config, ** args):
    # make sure you run fetch_results() command before this using this function
    with_config(config)
    work_dir = os.path.join(env.local_results, template(env.job_name_template))
    json_file = rearrange_files(work_dir)
    #load campaign
    covidsim_campaign = uq.Campaign(state_file=json_file, work_dir=work_dir)    

    print('========================================================')
    print('Reloaded campaign', covidsim_campaign.campaign_dir.split('/')[-1])
    print('========================================================')

    covidsim_campaign.collate()

    # Return dataframe containing all collated results
    collation_result = covidsim_campaign.get_collation_result()
    collation_result.to_csv(env.local_results + '/' +
                            template(env.job_name_template) +
                            '/collation_result.csv',
                            index=False
                            )
    print(collation_result)

    #load sampler
    sampler = covidsim_campaign.get_active_sampler()
    sampler.load_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_sampler_state.pickle"))
    covidsim_campaign.set_sampler(sampler)

    #load analysis
    analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
    analysis.load_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_analysis_state.pickle"))

    #compute the error at all admissible points, select direction with
    #highest error and add that direction to the grid
    analysis.adapt_dimension('f', collation_result)

    campaign.save_state(os.path.join(
        campaign.campaign_dir, "covid_easyvvuq_state.json"))

    analysis.save_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_sampler_state.pickle"))

@task
def analyse_adaptive_covid_easyvvuq(config, ** args):
    # make sure you run fetch_results() command before this using this function
    '''
    fab eagle_vecma analyse_covid_easyvvuq:UK_easyvvuq_test
    '''
    with_config(config)

    work_dir = os.path.join(env.local_results, template(env.job_name_template))

    json_file = rearrange_files(work_dir)
    # load campaign
    covidsim_campaign = uq.Campaign(state_file=json_file, work_dir=work_dir)

    # Combine the output from all runs associated with the current app
    covidsim_campaign.collate()

    # Return dataframe containing all collated results
    collation_result = covidsim_campaign.get_collation_result()
    collation_result.to_csv(env.local_results + '/' +
                            template(env.job_name_template) +
                            '/collation_result.csv',
                            index=False
                            )
    print(collation_result)

    # Post-processing analysis
    covidsim_analysis = uq.analysis.SCAnalysis(
        sampler=covidsim_campaign._active_sampler,
        qoi_cols=output_columns
    )

    covidsim_campaign.apply_analysis(covidsim_analysis)

    #this is a temporary subroutine which saves the entire state of
    #the analysis in a pickle file. The proper place for this is the database
    covidsim_analysis.save_state(os.path.join(
        covidsim_campaign.campaign_dir, "covid_easyvvuq_analysis_state.pickle"))
    
    # results = covidsim_campaign.get_last_analysis()
    # mu = results['statistical_moments']['cumDeath']['mean']
    # std = results['statistical_moments']['cumDeath']['std']

    # covid_analysis_add_file = env.local_results + '/' + \
    #     template(env.job_name_template) + '/covid_analysis.txt'

    # plot_grid(covidsim_analysis, ['Household_level_compliance_with_quarantine',
    #                               'Symptomatic_infectiousness_relative_to_asymptomatic'])


def rearrange_files(work_dir):
    # walk through all files in results_dir
    for root, _, _ in os.walk(os.path.join(work_dir, 'RUNS')):
        try:
            json_file = glob.glob(os.path.join(
                root, "covid_easyvvuq_state.json"))[0]
            state_folder = os.path.basename(root)
        except IndexError:
            pass

    # read json file
    with open(json_file, "r") as infile:
        json_data = json.load(infile)

    # create easyvvuq folder
    easyvvuq_folder = os.path.join(work_dir, json_data['campaign_dir'])
    try:
        # delete it is exists
        if os.path.exists(easyvvuq_folder):
            rmtree(easyvvuq_folder)
        os.mkdir(easyvvuq_folder)
        os.mkdir(os.path.join(easyvvuq_folder, 'runs'))
    except OSError:
        print("Creation of the directory %s failed" % easyvvuq_folder)
    else:
        print("Successfully created the directory %s " % easyvvuq_folder)

    # copy required files (*.xls) to easyvvuq_folder
    # in the results folder. xls files are in /RUNS/Run_1/output_dir/*.xls
    # and in easyvvuq_folder should be copied in /easyvvuq_folder/Run_1/*.xls
    # the code may seems to long, but I think it wroth to have clean and clear
    # easyvvuq_folder, and only copy files that are needed for analysis there :)
    #
    # local("rsync -r --include '*/' --include='*.xls' --exclude='*' {}/RUNS/
    # {}/runs/".format(work_dir, easyvvuq_folder))
    #
    db_copied = False
    json_copied = False
    print('Start copying csv files ...')
    for root, dirs, files in os.walk(os.path.join(work_dir, 'RUNS')):
        for file in files:
            if file.endswith(".severity.csv"):
                src_f = root
                des_f = os.path.join(easyvvuq_folder,
                                     'runs',
                                     root.split('/RUNS/')[1].split('/')[0])
                if not os.path.exists(des_f):
                    os.makedirs(des_f)

                with hide('output', 'running', 'warnings'), settings(warn_only=True):
                    local("cp %s %s" % (os.path.join(src_f, file),
                                        os.path.join(des_f, file)), capture=True)

            elif file == "campaign.db" and db_copied is False:
                # copy db file
                copyfile(os.path.join(root, file),
                         os.path.join(easyvvuq_folder,
                                      'campaign.db'))
                db_copied = True
            elif file == "covid_easyvvuq_state.json" and json_copied is False:
                # copy db file
                copyfile(os.path.join(root, file),
                         os.path.join(easyvvuq_folder,
                                      'covid_easyvvuq_state.json'))
                json_copied = True
    print('copying finished ...')
    # change database location file name in json file
    json_data['db_location'] = "sqlite:///" + \
        os.path.join(work_dir, json_data['campaign_dir'], 'campaign.db')
    # save json file
    json_file = os.path.join(easyvvuq_folder, 'covid_easyvvuq_state.json')
    with open(json_file, "w") as outfile:
        json.dump(json_data, outfile, indent=4)

    # updating db file
    engine = create_engine(json_data['db_location'])

    with engine.connect() as con:

        sql_cmd = "UPDATE app "
        sql_cmd += "SET output_decoder = JSON_SET(output_decoder,'$.state.target_filename','%s')" % (
            os.path.join('United_Kingdom_PC7_CI_HQ_SD_R0=2.4.avNE.severity.csv'))
        result = con.execute(sql_cmd)
        result.close()
        #
        # update run_dir
        # from /tmp/covid2yqcgs0w/runs/Run_1
        # to <FabSim_results>/UK_easyvvuq_test_eagle_vecma_28/Run_3/runs/Run_1
        sql_cmd = "UPDATE run "
        sql_cmd += "SET run_dir = '%s/'||run_name" % (
            os.path.join(work_dir, json_data['campaign_dir'], 'runs'))
        result = con.execute(sql_cmd)
        result.close()

        # update campaign_dir [/tmp/covid2yqcgs0w] ->
        # update runs_dir [/tmp/covid2yqcgs0w/runs] ->
        sql_cmd = "UPDATE campaign_info "
        sql_cmd += "SET campaign_dir='%s' , runs_dir='%s'" % (
            os.path.join(work_dir, json_data['campaign_dir']),
            os.path.join(work_dir, json_data['campaign_dir'], 'runs')
        )
        result = con.execute(sql_cmd)
        result.close()

    # return json_file address
    return json_file


# def plot_grid(covid_analysis, keys):

#     # find index of input keys in sampler.var
#     if isinstance(keys, str):
#         keys = [keys]
#     key_indexs = []
#     for key_value in keys:
#         key_indexs.append(
#             list(covid_analysis.sampler.vary.get_keys()).index(key_value))

#     if len(key_indexs) == 1:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, xlabel='RUNs number', ylabel=keys[0])
#         ax.plot(covid_analysis.xi_d[:, key_indexs[0]], 'ro')

#     if len(key_indexs) == 2:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, xlabel=keys[0], ylabel=keys[1])
#         ax.plot(covid_analysis.xi_d[:, key_indexs[0]],
#                 covid_analysis.xi_d[:, key_indexs[1]],
#                 'ro'
#                 )
#     elif len(key_indexs) == 3:
#         from mpl_toolkits.mplot3d import Axes3D
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d', xlabel=keys[0],
#                              ylabel=keys[1], zlabel=keys[2])
#         ax.scatter(covid_analysis.xi_d[:, key_indexs[0]],
#                    covid_analysis.xi_d[:, key_indexs[1]],
#                    covid_analysis.xi_d[:, key_indexs[2]]
#                    )
#     else:
#         print('Will only plot for N = 2 or N = 3.')

#     plt.tight_layout()
#     plt.show()


@task
def init_adaptive_covid_easyvvuq(** args):
    '''
    fab eagle_vecma run_covid_easyvvuq:UK_easyvvuq_test
    '''
    campaign, sampler = init_covid_campaign()

    campaign.save_state(os.path.join(
        campaign.campaign_dir, "covid_easyvvuq_state.json"))

    #this is a temporary subroutine which saves the entire state of
    #the sampler in a pickle file. The proper place for this is the database
    sampler.save_state(os.path.join(
        campaign.campaign_dir, "covid_easyvvuq_sampler_state.pickle"))

@task
def run_adaptive_covid_easyvvuq(config, ** args):
    # copy generated run folders to SWEEP directory
    # clean SWEEP, this part will be added to FabSim3.campaign2ensemble
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/SWEEP/*" % (sweep_dir))
    campaign2ensemble(config, campaign_dir=campaign.campaign_dir)

    # copy campaign.db and covid_easyvvuq_state.json to config folder
    # cp %s/*.{db,json} %s/  not working !!! with run function !!!
    local("cp %s/*.db %s/*.json %s/" % (campaign.campaign_dir,
                                        campaign.campaign_dir,
                                        path_to_config))
    CovidSim_ensemble(config, **args)
