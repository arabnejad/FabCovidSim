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

import fabsim3_cmd_api as fab

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
work_dir = '/home/wouter/VECMA/Campaigns'
config = 'UK_easyvvuq_test'

json_file = rearrange_files(work_dir)
# load campaign
campaign = uq.Campaign(state_file=json_file, work_dir=work_dir)

# Combine the output from all runs associated with the current app
campaign.collate()

# Return dataframe containing all collated results
collation_result = campaign.get_collation_result()
collation_result.to_csv(env.local_results + '/' +
                        template(env.job_name_template) +
                        '/collation_result.csv',
                        index=False
                        )
print(collation_result)

# Post-processing analysis
analysis = uq.analysis.SCAnalysis(
    sampler=campaign._active_sampler,
    qoi_cols=output_columns
)

campaign.apply_analysis(analysis)

#this is a temporary subroutine which saves the entire state of
#the analysis in a pickle file. The proper place for this is the database
analysis.save_state(os.path.join(campaign.campaign_dir, "covid_analysis_state.pickle"))