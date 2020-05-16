# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabCovidsim.
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
    env.script = 'CovidSim'
    path_to_config = find_config_file_path(config)
    print("local config file path at: %s" % path_to_config)
    sweep_dir = path_to_config + "/SWEEP"

    run_ensemble(config, sweep_dir, **args)

from plugins.FabCovidsim.covid_easyvvuq import run_covid_easyvvuq
from plugins.FabCovidsim.covid_easyvvuq import analyse_covid_easyvvuq
