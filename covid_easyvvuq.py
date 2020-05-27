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

tmp_dir = '/tmp/'
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
#output_columns = ["cumDeath"]
output_columns = ["R",
                  "SARI", "incSARI", "cumSARI",
                  "Critical", "incCritical", "cumCritical",
                  "incDeath", "cumDeath",
                  "incDeath_SARI", "incDeath_Critical",
                  "cumDeath_SARI", "cumDeath_Critical"]


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


class CustomEncoder(uq.encoders.GenericEncoder, encoder_name='CustomEncoder'):
    def encode(self, params={}, target_dir=''):
        """
        # Logistic curve for mortality
        k = params["mortality_k"]
        x0 = params["mortality_x0"]

        age = np.arange(5,90,5)
        curve = 1 / (1 + e**(-k*(age-x0)))
        """
        # scale default values found in pre param file
        default_mortality = np.array([0,
                                      1.60649128,
                                      2.291051747,
                                      2.860938008,
                                      3.382077741,
                                      3.880425012,
                                      4.37026577,
                                      4.861330415,
                                      5.361460943,
                                      5.877935626,
                                      6.4183471,
                                      6.991401405,
                                      7.607881726,
                                      8.282065409,
                                      9.034104744,
                                      9.894486491,
                                      10.91341144,
                                      12.18372915,
                                      13.9113346,
                                      16.74394356,
                                      22.96541429])
        curve = default_mortality * params["mortality_factor"]
        params["mortality_curve"] = curve

        proportion_symptomatic = [params["p_symptomatic"]] * 17
        params["Proportion_symptomatic"] = proportion_symptomatic

        super().encode(params, target_dir)


@task
def initialize_covidsim_campaign():

    # Set up a fresh campaign called "cannon"
    campaign = uq.Campaign(name='covid-sim', work_dir=tmp_dir)

    # Define parameter space for the cannonsim app
    params = json.load(open(get_plugin_path(
        "FabCovidsim") + '/templates/params_p_PC_CI_HQ_SD.json'))
    
    params["mortality_factor"] = {"default": 1,"type": "float"}
    
    params["p_symptomatic"] = {"default": 0.5,"type": "float"}

    # Create an encoder and decoder
    directory_tree = {'param_files': None}

    multiencoder = uq.encoders.MultiEncoder(
        uq.encoders.DirectoryBuilder(tree=directory_tree),
        CustomEncoder(
            template_fname=get_plugin_path("FabCovidsim") +
            '/templates/template_preUK_R0=2.0.txt',
            delimiter='$',
            target_filename='param_files/preUK_R0=2.0.txt'
        ),
        uq.encoders.GenericEncoder(
            template_fname=get_plugin_path("FabCovidsim") +
            '/templates/template_p_PC_CI_HQ_SD.txt',
            delimiter='$',
            target_filename='param_files/p_PC_CI_HQ_SD.txt'
        )
    )

    decoder = uq.decoders.SimpleCSV(target_filename='???', 
                                    output_columns=output_columns, 
                                    header=0,
                                    delimiter='\t')

    collater = uq.collate.AggregateSamples(average=False)

    # Add the app
    campaign.add_app(name="covid_PC_CI_HQ_SD",
                     params=params,
                     encoder=multiencoder,
                     collater=collater,
                     decoder=decoder)

    # Set the active app to be cannonsim (this is redundant when only one app
    # has been added)
    campaign.set_app("covid_PC_CI_HQ_SD")

    # Create a collation element for this campaign
    vary = {
        "Symptomatic_infectiousness_relative_to_asymptomatic": cp.Uniform(1.0, 2.5),
        "p_symptomatic": cp.Uniform(0.5, 0.9),
        "Latent_period": cp.Uniform(3.0, 7.0), # days
        "mortality_factor": cp.Uniform(0.8, 1.2),
    }
    sampler = uq.sampling.SCSampler(vary=vary,
                                    polynomial_order=1,
                                    quadrature_rule="G"
                                    )

    campaign.set_sampler(sampler)
    campaign.draw_samples()
    campaign.populate_runs_dir()

    campaign.save_state(os.path.join(campaign.campaign_dir,
                                     "campaign_state.json")
                        )
    sampler.save_state(os.path.join(campaign.campaign_dir,
                                    "campaign_sampler.pickle")
                       )

    return campaign.campaign_dir


@task
def run_covid_easyvvuq_standard(config, ** args):
    '''
    fab eagle_vecma run_covid_easyvvuq_standard:UK_easyvvuq_test
    '''
    campaign_dir = initialize_covidsim_campaign()

    # copy generated run folders to SWEEP directory
    # clean SWEPP, this part will be added to FabSim3.campaign2ensemble
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/SWEEP/*" % (sweep_dir))
    campaign2ensemble(config, campaign_dir=campaign_dir)

    # copy campaign.db, campaign_state.json, and campaign_sampler.pickle
    # to config folder
    # cp %s/*.{db,json} %s/  not working !!! with run function !!!
    local("cp %s/*.db %s/*.json %s/*.pickle %s/" % (campaign_dir,
                                                    campaign_dir,
                                                    campaign_dir,
                                                    path_to_config)
          )

    CovidSim_ensemble(config, **args)


@task
def analyse_covid_easyvvuq_standard(config, ** args):
    # make sure you run fetch_results() command before this using this function
    '''
    fab eagle_vecma analyse_covid_easyvvuq_standard:UK_easyvvuq_test
    '''

    with_config(config)

    work_dir = os.path.join(env.local_results, template(env.job_name_template))

    campaign_dir = rearrange_files(work_dir)
    json_file = os.path.join(campaign_dir, 'campaign_state.json')
    with open(json_file) as infile:
        json_data = json.load(infile)

    pprint(json_data)
    print('=' * 50)
    print(' ' * 5, 'Reloading campaign ', json_data["campaign_dir"])
    print('=' * 50)
    # load campaign
    campaign = uq.Campaign(state_file=json_file, work_dir=work_dir)

    # these loading sampler may be delete in next release
    sampler = campaign.get_active_sampler()
    sampler.load_state(os.path.join(campaign_dir, 'campaign_sampler.pickle'))
    campaign.set_sampler(sampler)

    # Combine the output from all runs associated with the current app
    campaign.collate()

    # Return dataframe containing all collated results
    collation_result = campaign.get_collation_result()

    collation_result.to_csv(os.path.join(campaign_dir, 'collation_result.csv'),
                            index=False
                            )

    print(collation_result)

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(sampler=campaign._active_sampler,
                                      qoi_cols=output_columns
                                      )
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    # this is a temporary subroutine which saves the entire state of
    # the analysis in a pickle file. The proper place for this is the database
    #analysis.save_state(os.path.join(campaign_dir, 'campaign_analysis.pickle'))
    '''
    output_columns = ["R",
                      "SARI", "incSARI", "cumSARI",
                      "Critical", "incCritical", "cumCritical",
                      "incDeath", "cumDeath",
                      "incDeath_SARI", "incDeath_Critical",
                      "cumDeath_SARI", "cumDeath_Critical"]
    '''
    #output_columns = ["incDeath", "SARI", "Critical", "R"]
    # --------------------------------------------------------------------------
    #                   Plotting
    # --------------------------------------------------------------------------

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
    plt.show()



def rearrange_files(work_dir):
    # walk through all files in results_dir
    for root, _, _ in os.walk(os.path.join(work_dir, 'RUNS')):
        try:
            json_file = glob.glob(os.path.join(
                root, "campaign_state.json"))[0]
            state_folder = os.path.basename(root)
            break
        except IndexError:
            pass

    # read json file
    with open(json_file, "r") as infile:
        json_data = json.load(infile)

    # create easyvvuq folder
    campaign_dir = os.path.join(work_dir, json_data['campaign_dir'])
    try:
        # delete it is exists
        if os.path.exists(campaign_dir):
            rmtree(campaign_dir)
        os.mkdir(campaign_dir)
    except OSError:
        print("Creation of the directory %s failed" % campaign_dir)
    else:
        print("Successfully created the directory %s " % campaign_dir)

    # copy campaign.db, campaign_state.json, and campaign_sampler.pickle
    # to campaign_dir
    print('Start copying *.{db, json, pickle} files ...')
    for root, dirs, files in os.walk(os.path.join(work_dir, 'RUNS')):
        for file in files:
            if file == "campaign.db":
                with hide('running'), settings(warn_only=True):
                    local("cp %s/*.db %s/*.json %s/*.pickle %s/" % (root,
                                                                    root,
                                                                    root,
                                                                    campaign_dir)
                          )
                break

    print('copying finished ...')

    # change database location file name in json file
    json_data['db_location'] = "sqlite:///" + \
        os.path.join(work_dir, json_data['campaign_dir'], 'campaign.db')
    # save json file
    json_file = os.path.join(campaign_dir, 'campaign_state.json')
    with open(json_file, "w") as outfile:
        json.dump(json_data, outfile, indent=4)

    # updating db file
    engine = create_engine(json_data['db_location'])

    with engine.connect() as con:

        sql_cmd = "UPDATE app "
        sql_cmd += "SET output_decoder = JSON_SET(output_decoder,'$.state.target_filename','%s')" % (
            os.path.join('United_Kingdom_PC_CI_HQ_SD_R0=2.4.avNE.severity.xls'))
        result = con.execute(sql_cmd)
        result.close()
        #
        # update run_dir
        #   /tmp/covid-sim23xhsa6n/runs/Run_1
        #   -->
        #   <FabSim_results>/UK_easyvvuq_test_eagle_vecma_28/Run_3/runs/Run_1
        sql_cmd = "UPDATE run "
        sql_cmd += "SET run_dir = '%s/'||run_name||'/%s'" % (
            os.path.join(work_dir, 'RUNS'), 'output_dir')
        result = con.execute(sql_cmd)
        result.close()

        # update campaign_dir
        #   [/tmp/covid-sim23xhsa6n]
        #   -->
        #   <FabSim_results>/UK_easyvvuq_test_eagle_vecma_28/covid-sim23xhsa6n
        # update runs_dir
        #   [/tmp/covid-sim23xhsa6n/runs]
        #   -->
        #   <FabSim_results>/UK_easyvvuq_test_eagle_vecma_28/RUNS
        sql_cmd = "UPDATE campaign_info "
        sql_cmd += "SET campaign_dir='%s' , runs_dir='%s'" % (
            os.path.join(work_dir, json_data['campaign_dir']),
            os.path.join(work_dir, 'RUNS')
        )
        result = con.execute(sql_cmd)
        result.close()

    # return campaign_dir path
    return campaign_dir

def plot_grid(covid_analysis, keys):

    # find index of input keys in sampler.var
    if isinstance(keys, str):
        keys = [keys]
    key_indexs = []
    for key_value in keys:
        key_indexs.append(
            list(covid_analysis.sampler.vary.get_keys()).index(key_value))

    if len(key_indexs) == 1:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel='RUNs number', ylabel=keys[0])
        ax.plot(covid_analysis.xi_d[:, key_indexs[0]], 'ro')

    if len(key_indexs) == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, xlabel=keys[0], ylabel=keys[1])
        ax.plot(covid_analysis.xi_d[:, key_indexs[0]],
                covid_analysis.xi_d[:, key_indexs[1]],
                'ro'
                )
    elif len(key_indexs) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', xlabel=keys[0],
                             ylabel=keys[1], zlabel=keys[2])
        ax.scatter(covid_analysis.xi_d[:, key_indexs[0]],
                   covid_analysis.xi_d[:, key_indexs[1]],
                   covid_analysis.xi_d[:, key_indexs[2]]
                   )
    else:
        print('Will only plot for N = 2 or N = 3.')

    plt.tight_layout()
    plt.show()
