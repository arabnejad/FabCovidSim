# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabCovid19.
#
# authors: Hamid Arabnejad, Derek Groen

from base.fab import *
import shutil
from os import makedirs, path, walk
import csv
from pprint import pprint

# Add local script, blackbox and template path.
add_local_paths("FabCovidsim")


@task
def CovidSim(config,
             memory='20GB',
             label="",
             **args):
    """
    Submit a COVID-19 CovidSim job to the remote queue.:
    fab eagle_vecma CovidSim:UK_sample<,memory=MemorySize><label=your_lable>

    fab eagle_vecma CovidSim:UK_sample,memory=20000

    """
    update_environment(args, {"output_dir": "output_dir",
                              "memory": memory
                              })
    if len(label) > 0:
        print("adding label: ", label)
        env.job_name_template += "_{}".format(label)

    with_config(config)
    print(args)
    execute(put_configs, config)
    job(dict(script='CovidSim'), args)


@task
def CovidSim_ensemble(config,
                      script,
                      memory='20GB',
                      label="",
                      **args):
    """
    Submits an ensemble of COVID-19 CovidSim jobs.
    One job is run for each file in /FabCovidSim/config_files/UK_sample/SWEEP.

    fab eagle_vecma CovidSim_ensemble:UK_sample
    fab qcg CovidSim_ensemble:UK_sample,PilotJob=True
    fab qcg CovidSim_ensemble:UK_sample,replicas=5,PilotJob=True
    fab eagle_vecma CovidSim_ensemble:UK_sample,PilotJob=True
    fab eagle_vecma CovidSim_ensemble:UK_sample,replicas=5
    """

    update_environment(args, {"output_dir": "output_dir",
                              "memory": memory
                              })
    if len(label) > 0:
        print("adding label: ", label)
        env.job_name_template += "_{}".format(label)
    # required by qcg-pj to distribute threads correctly
    env.task_model = 'threads'
    env.script = script
    path_to_config = find_config_file_path(config)
    print("local config file path at: %s" % path_to_config)
    sweep_dir = path_to_config + "/SWEEP"

    run_ensemble(config, sweep_dir, **args)
    

@task
def run_adaptive_easyvvuq(config, 
                          campaign_dir, 
                          script = 'CovidSim',
                          skip=0, 
                          ** args):
    # copy generated run folders to SWEEP directory
    # clean SWEEP, this part will be added to FabSim3.campaign2ensemble
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/*" % (sweep_dir))
    campaign2ensemble(config, campaign_dir=campaign_dir)

    #option to remove earlier runs from the sweep dir
    if int(skip) > 0:
        for i in range(int(skip)):
            local('rm -r %s/Run_%s' %(sweep_dir, i+1))

    CovidSim_ensemble(config, script, **args)
    
@task
def get_adaptive_easyvvuq(config,
                          campaign_dir,
                          number_of_samples, **args):
    """
    Fetches sample output from host, copies results to adaptive EasyVVUQ work directory
    """
    
    fetch_results()

    #loop through all result dirs to find result dir of sim_ID
    found = False
    dirs = os.listdir(env.local_results)
    for dir_i in dirs:
        #We are assuming here that the name of the directory with the runs dirs
        #STARTS with the config name. e.g. <config_name>_eagle_vecma_28 and
        #not PJ_header_<config_name>_eagle_vecma_28
        if config == dir_i[0:len(config)]:
            found = True
            break

    if found:
        #This compies the entire result directory from the (remote) host back to the
        #EasyVVUQ Campaign directory
        print('Copying results from', env.local_results + '/' + dir_i + 'to' + campaign_dir)
        ensemble2campaign(env.local_results + '/' + dir_i, campaign_dir, **args)
        
        #If the same FabSim3 config name was used before, the statement above
        #might have copied more runs than currently are used by EasyVVUQ.
        #This removes all runs in the EasyVVUQ campaign dir (not the Fabsim results dir) 
        #for which Run_X with X > number of current samples.
        dirs = os.listdir(os.path.join(campaign_dir, 'runs'))
        for dir_i in dirs:
            run_ID = int(dir_i.split('_')[-1])
            if run_ID > int(number_of_samples):
                local('rm -r %s/runs/Run_%d' % (campaign_dir, run_ID))
                print('Removing Run %d from %s/runs' % (run_ID, campaign_dir))
                
    else:
        print('Campaign dir not found')
    
