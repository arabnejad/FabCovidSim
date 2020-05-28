# FabSim3 Commands Python API
#
# This file maps command-line instructions for FabSim3 to Python functions.
# NOTE: No effort is made to map output back to FabSim, as this complicates
# the implementation greatly.
#
# This file can be included in any code base. 
# It has no dependencies, but does require a working FabSim3 installation.

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

def fabsim(command, arguments, machine = 'localhost'):
    """
    Generic function for running any FabSim3 command.
    """
    print('Executing', "fab {} {}:{}".format(machine, command, arguments))
    os.system("fab {} {}:{}".format(machine, command, arguments))

def run_uq_ensemble(config, campaign_dir, machine='localhost', skip = 0, **args):
    """
    Launches a UQ ensemble.
    """
    # sim_ID = campaign_dir.split('/')[-1]
    arguments = "{},campaign_dir={},skip={}".format(config, campaign_dir, skip)
    fabsim("run_adaptive_covid_easyvvuq", arguments, machine=machine)
    
def get_uq_samples(config, campaign_dir, machine = 'localhost'):
    """
    Retrieves results from UQ ensemble
    """
    # sim_ID = campaign_dir.split('/')[-1]
    arguments = "{},campaign_dir={}".format(config, campaign_dir)
    fabsim("get_uq_samples", arguments, machine=machine)
    
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
