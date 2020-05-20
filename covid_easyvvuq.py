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

class CustomEncoder(uq.encoders.JinjaEncoder, encoder_name='CustomEncoder'):
    def encode(self, params={}, target_dir='', fixtures=None):
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

        super().encode(params, target_dir, fixtures)




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
def init_covid_campaign():

    # Set up a fresh campaign called "cannon"
    my_campaign = uq.Campaign(name='covid', work_dir=tmp_dir)

    # Define parameter space for the cannonsim app
    params = json.load(open(get_plugin_path(
        "FabCovidsim") + '/templates/params.json'))
    params["mortality_factor"] = {
                                    "default": 1,
                                    "type": "float"
                                }
    params["p_symptomatic"] = {
                                      "default": 0.5,
                                      "type": "float"
                                  }

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
                        params=params,
                        encoder=multiencoder_p_PC7_CI_HQ_SD,
                        collater=collater,
                        decoder=decoder)
    # Set the active app to be cannonsim (this is redundant when only one app
    # has been added)
    my_campaign.set_app("covid_p_PC7_CI_HQ_SD")

    # Create a collation element for this campaign
    vary = {
        "Symptomatic_infectiousness_relative_to_asymptomatic": cp.Uniform(1.0, 2.5),
        "p_symptomatic": cp.Uniform(0.1, 0.9),
        "Latent_period": cp.Uniform(2.0, 16.0), # days
        "mortality_factor": cp.Uniform(0.5, 1.5),
    }
    my_sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1)

    my_campaign.set_sampler(my_sampler)
    my_campaign.draw_samples()

    my_campaign.populate_runs_dir()

    return my_campaign


@task
def analyse_covid_easyvvuq(config, ** args):
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

    results = covidsim_campaign.get_last_analysis()
    mu = results['statistical_moments']['cumDeath']['mean']
    std = results['statistical_moments']['cumDeath']['std']

    covid_analysis_add_file = env.local_results + '/' + \
        template(env.job_name_template) + '/covid_analysis.txt'

    plot_grid(covidsim_analysis, ['Household_level_compliance_with_quarantine',
                                  'Symptomatic_infectiousness_relative_to_asymptomatic'])


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


@task
def run_covid_easyvvuq(config, ** args):
    '''
    fab eagle_vecma run_covid_easyvvuq:UK_easyvvuq_test
    '''
    campaign = init_covid_campaign()

    campaign.save_state(os.path.join(
        campaign.campaign_dir, "covid_easyvvuq_state.json"))

    # copy generated run folders to SWEEP directory
    # clean SWEPP, this part will be added to FabSim3.campaign2ensemble
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
