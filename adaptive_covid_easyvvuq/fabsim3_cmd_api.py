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

def fabsim(command, arguments, machine = 'localhost'):
    """
    Generic function for running any FabSim3 command.
    """
    print('Executing', "fab {} {}:{}".format(machine, command, arguments))
    os.system("fabsim {} {}:{}".format(machine, command, arguments))

def fetch_results(machine='localhost'):
    fabsim("fetch_results", "", machine)
    
def resubmit_ensemble(config, command='CovidSim_ensemble', machine='localhost', 
                      PilotJob=False):
    arguments = "{},PilotJob={}".format(config, PilotJob)
    fabsim(command, arguments, machine)

def run_uq_ensemble(config, campaign_dir, script, machine='localhost', skip = 0,
                    PilotJob = False, **args):
    """
    Launches a UQ ensemble.
    """
    # sim_ID = campaign_dir.split('/')[-1]
    arguments = "{},campaign_dir={},script={},skip={},PilotJob={}".format(config, campaign_dir, script, skip, PilotJob)
    fabsim("run_adaptive_easyvvuq", arguments, machine=machine)
    
def get_uq_samples(config, campaign_dir, number_of_samples, machine = 'localhost'):
    """
    Retrieves results from UQ ensemble
    """
    # sim_ID = campaign_dir.split('/')[-1]
    arguments = "{},campaign_dir={},number_of_samples={}".format(config, campaign_dir, number_of_samples)
    fabsim("get_adaptive_easyvvuq", arguments, machine=machine)
    
# def rearrange_files(config):
#     arguments = "{}".format(config)
#     fabsim("rearrange_files", arguments, machine="localhost")