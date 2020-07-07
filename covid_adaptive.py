# -*- coding: utf-8 -*-
#
# This source file is part of the FabSim software toolkit, which is distributed under the BSD 3-Clause license.
# Please refer to LICENSE for detailed information regarding the licensing.
#
# This file contains FabSim definitions specific to FabCovidsim.
#
# authors: Hamid Arabnejad, Wouter Edeling, Diana Suleimenova, Robert Sinclair
import matplotlib
matplotlib.use('Agg')
import numpy as np
import easyvvuq as uq
import chaospy as cp
import os
import pathlib
import sys
import pytest
from time import sleep
import math
from pprint import pprint
import subprocess
import json
from shutil import copyfile, rmtree
from sqlalchemy import create_engine
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from matplotlib import gridspec

#-------------------------------------------------------------------------
#                           Global Variables
#-------------------------------------------------------------------------
output_filename = "United_Kingdom_PC_CI_HQ_SD_R0=2.4.avNE.severity.xls"
target_filename = os.path.join("output_dir", output_filename)


#-------------------------------------------------------------------------


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
    fab eagle_vecma covid_init:GB_suppress,output_column=cumDeath,extra_lable=''
    fab eagle_vecma covid_analyse:GB_suppress,output_column=cumDeath,extra_lable=''
    loop
        fab eagle_vecma covid_look_ahead:GB_suppress,output_column=cumDeath,extra_lable=''
        fab eagle_vecma covid_adapt:GB_suppress,output_column=cumDeath,extra_lable=''

'''


'''
output_columns = ["R",
                  "SARI", "incSARI", "cumSARI",
                  "Critical", "incCritical", "cumCritical",
                  "incDeath", "cumDeath",
                  "incDeath_SARI", "incDeath_Critical",
                  "cumDeath_SARI", "cumDeath_Critical"]
'''
output_column = "cumDeath"


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
def covid_all_together(config, output_column="cumDeath", extra_lable='', repeat=1, ** args):
    '''
    ╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                                                                 ║
    ║ fabsim eagle_vecma covid_all_together:GB_suppress,output_column=cumDeath,extra_lable='',repeat=5 2>&1 | tee execution_log.txt   ║
    ║                                                                                                                                 ║
    ╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    '''
    covid_init(config, output_column, extra_lable, ** args)

    covid_analyse(config, output_column, extra_lable, ** args)
    for i in range(1, repeat + 1):
        covid_look_ahead(config, output_column, extra_lable, ** args)
        covid_adapt(config, output_column, extra_lable, ** args)


@task
def covid_adapt_RAMP_report(config, ** args):
    '''
    fab eagle_vecma covid_adapt_RAMP_report:GB_suppress
    '''
    work_dir_adapt = os.path.join(os.path.dirname(__file__),
                                  'covid_adaptive_test')
    backup_dir = os.path.join(work_dir_adapt, 'backup_RAMP')

    load_campaign_files(work_dir_adapt, backup_dir)

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
    analysis = uq.analysis.SCAnalysis(
        sampler=sampler, qoi_cols=[output_column])
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    results_pickle_file = os.path.join(work_dir_adapt,
                                       "results_campaign_get_last_analysis.pickle")

    if not path.isfile(results_pickle_file):
        print('========================================================')
        print('campaign apply_analysis and save results into pickle file')
        print('========================================================')

        mean, D, S_u = analysis.get_pce_sobol_indices('cumDeath', 'all')

        campaign.apply_analysis(analysis)
        results = campaign.get_last_analysis()

        # save results array with pickle
        file = open(results_pickle_file, 'wb')
        pickle.dump(results, file)
        file.close()

        file = open(os.path.join(work_dir_adapt,
                                 "mean_get_pce_sobol_indices.pickle"), 'wb')
        pickle.dump(mean, file)
        file.close()

        file = open(os.path.join(work_dir_adapt,
                                 "D_get_pce_sobol_indices.pickle"), 'wb')
        pickle.dump(D, file)
        file.close()

        file = open(os.path.join(work_dir_adapt,
                                 "S_u_get_pce_sobol_indices.pickle"), 'wb')
        pickle.dump(S_u, file)
        file.close()
    else:
        print('========================================================')
        print('load results from pickle file')
        print('========================================================')

        # load results from pickle
        file = open(results_pickle_file, 'rb')
        results = pickle.load(file)
        file.close()

        file = open(os.path.join(work_dir_adapt,
                                 "mean_get_pce_sobol_indices.pickle"), 'rb')
        mean = pickle.load(file)
        file.close()

        file = open(os.path.join(work_dir_adapt,
                                 "D_get_pce_sobol_indices.pickle"), 'rb')
        D = pickle.load(file)
        file.close()

        file = open(os.path.join(work_dir_adapt,
                                 "S_u_get_pce_sobol_indices.pickle"), 'rb')
        S_u = pickle.load(file)
        file.close()

    '''
    "Proportion_of_places_remaining_open_after_closure_by_place_type2": cp.Uniform(0.2, 0.7),
    "Proportion_of_places_remaining_open_after_closure_by_place_type3": cp.Uniform(0.7, 1.0),
    "Residual_place_contacts_after_household_quarantine_by_place_type0": cp.Uniform(0.2, 0.3),
    "Relative_place_contact_rate_given_social_distancing_by_place_type0": cp.Uniform(0.8, 1.0),
    "Relative_place_contact_rate_given_social_distancing_by_place_type1": cp.Uniform(0.5, 0.8),
    "Relative_rate_of_random_contacts_if_symptomatic": cp.Uniform(0.4, 0.6),
    "Relative_level_of_place_attendance_if_symptomatic0": cp.Uniform(0.2, 0.3),
    "Relative_level_of_place_attendance_if_symptomatic2": cp.Uniform(0.4, 0.6),
    "CLP1": cp.Uniform(60, 400),
    '''
    defaults = {
        "Proportion_of_places_remaining_open_after_closure_by_place_type2": 0.25,
        "Proportion_of_places_remaining_open_after_closure_by_place_type3": 1.0,
        "Residual_place_contacts_after_household_quarantine_by_place_type0": 0.25,
        "Relative_place_contact_rate_given_social_distancing_by_place_type0": 1.0,
        "Relative_place_contact_rate_given_social_distancing_by_place_type1": 0.75,
        "Relative_rate_of_random_contacts_if_symptomatic": 0.5,
        "Relative_level_of_place_attendance_if_symptomatic0": 0.25,
        "Relative_level_of_place_attendance_if_symptomatic2": 0.5
    }

    choice = {
        "Proportion_of_places_remaining_open_after_closure_by_place_type2": "Uniform(0.2, 0.7)",
        "Proportion_of_places_remaining_open_after_closure_by_place_type3": "Uniform(0.7, 1.0)",
        "Residual_place_contacts_after_household_quarantine_by_place_type0": "Uniform(0.2, 0.3)",
        "Relative_place_contact_rate_given_social_distancing_by_place_type0": "Uniform(0.8, 1.0)",
        "Relative_place_contact_rate_given_social_distancing_by_place_type1": "Uniform(0.5, 0.8)",
        "Relative_rate_of_random_contacts_if_symptomatic": "Uniform(0.4, 0.6)",
        "Relative_level_of_place_attendance_if_symptomatic0": "Uniform(0.2, 0.3)",
        "Relative_level_of_place_attendance_if_symptomatic2": "Uniform(0.4, 0.6)"
    }

    params = ['Proportion of places remaining open after closure by place type [universities]',
              'Proportion of places remaining open after closure by place type [workplaces]',
              'Residual place contacts after household quarantine by place type [all]',
              'Relative place contact rate given social distancing by place type [elementary school, high schools]',
              'Relative place contact rate given social distancing by place type [universities, workplaces]',
              'Relative rate of random contacts if symptomatic',
              'Relative level of place attendance if symptomatic [elementary school, high schools]',
              'Relative level of place attendance if symptomatic [universities, workplaces]'
              ]

    adapt_measure = np.max(analysis.l_norm, axis=0)

    data = {
        'default': [],
        'min': [],
        'max': [],
        'reason for range': [],
        'max quadrature order': [],
    }
    index = []

    for i, (key, dist) in enumerate(analysis.sampler.vary.vary_dict.items()):
        data['min'] += [dist.lower]
        data['max'] += [dist.upper]
        data['max quadrature order'] += [adapt_measure[i]]
        data['default'] += [defaults[key]]
        data['reason for range'] += [choice[key]]
    data['min'] = [float(x) for x in data['min']]
    data['max'] = [float(x) for x in data['max']]

    # for var, dist in analysis.sampler.vary.vary_dict.items():
    #    data['min'] += [dist.]

    data = pd.DataFrame(data, index=params)
    sorted_data = data.sort_values('max quadrature order', ascending=False)

    print(adapt_measure)
    pprint(sorted_data)

    # plot first order sobol indeces by time
    sobols_first = results["sobols_first"][output_column]

    plt.figure(figsize=(9.0, 8.2))

    x = range(801)
    for i, v in enumerate(sobols_first):

        y = sobols_first[v]
        important = False
        # if y[-1] > 0.001:
        ax = sns.lineplot(x=x, y=sobols_first[v], label=params[i])

    '''
    top=0.768,
    bottom=0.082,
    left=0.07,
    right=0.972,
    hspace=0.2,
    wspace=0.2
    '''
    plt.legend(bbox_to_anchor=(0., 1.3),
               loc="upper left", borderaxespad=0.)
    plt.xlabel('Days')
    plt.ylabel('First order Sobol index')
    plt.tight_layout()

    plt.savefig(os.path.join(work_dir_adapt,
                             'RAMP_report_First_order_Sobol_index.pdf'))
    plt.close()

    # plot total sobal index for each variable
    plt.clf()
    plt.figure(figsize=(9.0, 8.2))
    sns.set_palette("colorblind")
    x = range(801)
    S_T = {i: np.zeros(801) for i in range(14)}

    for key in S_u.keys():
        for index in key:
            S_T[index] += S_u[key]

    for i, label in enumerate(sobols_first):
        y = S_T[i]
        # if y[-1] > 0.02:
        sns.lineplot(x=x, y=y, label=params[i])

    plt.legend(bbox_to_anchor=(0., 1.3),
               loc="upper left", borderaxespad=0.)
    plt.xlabel('Days')
    plt.ylabel('Total Sobol index')
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir_adapt,
                             'RAMP_report_Total_Sobol_index.pdf'))

    plt.close()

    # this is useful to check if there are some significant high order
    # interactions
    for key in S_u.keys():
        if S_u[key][-1] > 0.01:
            print(key, S_u[key][-1])

    print('\n key:')
    for i, label in enumerate(sobols_first):
        print(i, label)

    lower1, upper1, lower2, upper2, total_deaths = get_confidence_intervals(
        analysis, 'cumDeath', 1000)

    fig = plt.figure(figsize=(8, 4))
    spec = gridspec.GridSpec(ncols=2, nrows=1,
                             width_ratios=[3, 1])

    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1], sharey=ax1)
    ax2.get_xaxis().set_ticks([])
    fig.subplots_adjust(wspace=0)
    plt.setp(ax2.get_yticklabels(), visible=False)

    ax1.fill_between(x, lower2, upper2, color='#eee9ff', label='95% CI')
    ax1.fill_between(x, lower1, upper1, color='#aa99cc', label='68% CI')
    mean = results["statistical_moments"][output_column]["mean"]
    ax1.plot(x, mean, label='Mean')
    ax1.legend(loc="upper left")

    ax1.set_xlabel('Days')
    ax1.set_ylabel('Cumulative deaths')
    ax2.set_xlabel('Frequency')
    # ax2.set_title('Total deaths distribution')
    ax2.axis('off')

    ax2 = sns.distplot(total_deaths, vertical=True)

    plt.savefig(os.path.join(work_dir_adapt,
                             'RAMP_report_statistical_moments_mean.pdf'))

    plt.show()

    exit()
    output_column = "cumDeath"
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="days", ylabel=output_column)
    mean = results["statistical_moments"][output_column]["mean"]
    std = results["statistical_moments"][output_column]["std"]
    ax.plot(mean)
    ax.plot(mean + std, '--r')
    ax.plot(mean - std, '--r')
    plt.tight_layout()
    plt.show()

    backup_campaign_files()


def get_confidence_intervals(self, qoi, n_samples, conf1=0.36, conf2=0.9):
    """
    Compute the confidence intervals based upon samples of the SC surrogate.
    Uses a non-parametric approach based upon the empirical cumulative
    distribution function.

    Parameters:
    -----------
    qoi (str): name of the Quantity of Interest for which to compute the intervals
    n_samples (int): number of surrogate samples to be used in the computation
    conf (float in [0, 1]): the confidence interval magnitude

    Returns:
    --------
    lower, upper (array of floats): the upper and lower bound of the interval
    """

    # lower bound = alpha, upper bound = 1 - alpha
    alpha1 = 0.5 * (1.0 - conf1)
    alpha2 = 0.5 * (1.0 - conf2)

    # draw n_samples draws from the input distributions
    xi_mc = np.zeros([n_samples, self.N])
    idx = 0
    for dist in self.sampler.vary.get_values():
        xi_mc[:, idx] = dist.sample(n_samples)
        idx += 1

    # sample the surrogate n_samples times
    surr_samples = np.zeros([n_samples, self.N_qoi])
    print('Sampling surrogate %d times' % (n_samples,))
    for i in range(n_samples):
        surr_samples[i, :] = self.surrogate(qoi, xi_mc[i])
        if np.mod(i, 100) == 0:
            print('%d of %d' % (i + 1, n_samples))
    print('done')

    # arrays for lower and upper bound of the interval
    lower1 = np.zeros(self.N_qoi)
    upper1 = np.zeros(self.N_qoi)
    lower2 = np.zeros(self.N_qoi)
    upper2 = np.zeros(self.N_qoi)

    # the probabilities of the ecdf
    prob = np.linspace(0, 1, n_samples)
    # the closest locations in prob that correspond to the interval bounds
    idx10 = np.where(prob <= alpha1)[0][-1]
    idx11 = np.where(prob <= 1.0 - alpha1)[0][-1]
    idx20 = np.where(prob <= alpha2)[0][-1]
    idx21 = np.where(prob <= 1.0 - alpha2)[0][-1]

    # for every location of qoi compute the ecdf-based confidence interval
    for i in range(self.N_qoi):
        # the sorted surrogate samples at the current location
        samples_sorted = np.sort(surr_samples[:, i])
        # the corresponding confidence interval
        lower1[i] = samples_sorted[idx10]
        upper1[i] = samples_sorted[idx11]
        lower2[i] = samples_sorted[idx20]
        upper2[i] = samples_sorted[idx21]

    total_deaths = surr_samples[:, -1]

    return lower1, upper1, lower2, upper2, total_deaths


@task
def covid_adapt(config, output_column="cumDeath", extra_lable='', ** args):
    '''
    fab eagle_vecma covid_adapt:GB_suppress
    '''
    '''
    ╔════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                    ║
    ║    fab eagle_vecma covid_adapt:GB_suppress,output_column=cumDeath,extra_lable=''   ║
    ║                                                                                    ║
    ╚════════════════════════════════════════════════════════════════════════════════════╝
    '''
    print('╔═════════════════════════════════╗')
    print('║           covid_adapt           ║')
    print('╚═════════════════════════════════╝')

    work_dir_adapt = os.path.join(os.path.dirname(__file__),
                                  'covid_adaptive_test')
    work_dir_adapt += '[%s]' % (output_column)
    if len(extra_lable) > 0:
        work_dir_adapt += extra_lable

    config_yaml_file = os.path.join(
        os.path.dirname(__file__),
        "covid_adaptive_test[%s]%s.yml" % (output_column, extra_lable)
    )

    load_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)

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
    analysis = uq.analysis.SCAnalysis(
        sampler=sampler, qoi_cols=[output_column])
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # fetch only the required folder from remote machine
    with_config(config)
    # env.job_name_template += "_{}".format(job_label)
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))

    # instead of checking job scheduler, every time_step
    # fetch results and check if we have all the required output files
    # which means all jobs are finished correctly
    time_step = 5  # minutes
    time_step_passed = 0
    while True:
        print("fetching results from remote machine ...")

        with hide('output', 'running'), settings(warn_only=True):
            fetch_results(regex=job_folder_name)

        print("Done\n")

        # copy only output folder into local campaign_dir :)
        src = os.path.join(env.local_results, job_folder_name, 'RUNS')
        des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')

        # if we decided to step backs to runs, we need to remove some runs from
        # fetched results and also from previous generated runs by easyvvuq
        remove_extra_runs(campaign.campaign_db._next_run, src, des)

        print("Syncing output_dir ...")
        with hide('output', 'running'):
            local(
                "rsync -av -m -v \
                --include='/*/' \
                --include='/*/output_dir/***'  \
                --exclude='*' \
                {}/  {} ".format(src, des)
            )
        print("Done\n")

        unfinished_jobs = []
        for i in range(1, campaign.campaign_db._next_run):
            file_to_check = os.path.join(des, 'Run_%d' % (i), target_filename)
            if not os.path.isfile(file_to_check):
                unfinished_jobs.append(file_to_check)

        print("number of unfinished_jobs = %d" % (len(unfinished_jobs)))
        if len(unfinished_jobs) == 0:
            break
        else:
            sleep(time_step * 60)

        print("time %s minutes of 60 minutes passed " % (time_step_passed))
        time_step_passed += time_step

        if time_step_passed > 60:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('something is not right . . .')
            print('jobs are submitted in interactive queue, and some of them ')
            print('are not finished yet, check jobs')
            print('list of unfinished_jobs')
            pprint(unfinished_jobs)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print('========================================================')
    print('All required output_files are fetched . . .')
    print('========================================================')

    campaign.collate()

    # compute the error at all admissible points, select direction with
    # highest error and add that direction to the grid
    data_frame = campaign.get_collation_result()
    print("len(data_frame.run_id.unique()) = %d" %
          (len(data_frame.run_id.unique())))

    analysis.adapt_dimension(output_column, data_frame)

    # save everything
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))
    analysis.save_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # apply analysis
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    params = list(analysis.sampler.vary.get_keys())
    for i, param in enumerate(params):
        params[i] = param.replace('_', ' ')

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

    '''
    fig = plt.figure(figsize=[12, 4])
    ax = fig.add_subplot(131, xlabel='days', ylabel=output_column,
                         title='Surrogate samples')
    ax.plot(analysis.get_sample_array(
        output_column).T, 'ro', alpha=0.5)
    '''

    sobols_first = results["sobols_first"][output_column]

    for param in sobols_first.keys():
        print(sobols_first[param][-1], param)

    sns.set_palette("colorblind")
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    plt.figure(figsize=(16, 7))
    x = range(801)

    for i, v in enumerate(sobols_first):
        y = sobols_first[v]
        important = False
        plt.plot(x, sobols_first[v], label=params[i],
                 linestyle=LINE_STYLES[i % NUM_STYLES],
                 linewidth=2.0)
        #print(LINE_STYLES[i % NUM_STYLES])
        #ax.lines[0].set_linestyle(LINE_STYLES[i % NUM_STYLES])

    '''
    for v in sobols_first:
        y = sobols_first[v]
        important = False
        if y[-1] != 0:
        sns.lineplot(x=x, y=sobols_first[v], label=v)
    '''
    leg = plt.legend(bbox_to_anchor=(1.04, 0.5),
                     loc="center left", borderaxespad=0.)
    leg_lines = leg.get_lines()
    for i, leg in enumerate(leg_lines):
        leg.set_linestyle(LINE_STYLES[i % NUM_STYLES])
    print(len(leg_lines))

    leg_lines[5].set_linestyle(":")

    plt.xlabel('Days')
    plt.ylabel('First order Sobol index')
    plt.tight_layout()
    '''
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Days')
    plt.ylabel('First order Sobol index')
    '''

    plt.savefig(os.path.join(work_dir_adapt,
                             'First_order_Sobol_index_%d[%s]' %
                             (sampler.number_of_adaptations,
                              output_column)
                             ),
                dpi=400)

    # generate n_mc samples from the input distributions
    '''
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

    ##################################
    # analysis.mean_history #
    ##################################
    plt.clf()
    ax = fig.add_subplot(111, xlabel='plot_analysis.mean_history.T')

    ax.plot(np.array(analysis.mean_history).T)

    plt.tight_layout()
    plt.savefig(os.path.join(work_dir_adapt, 'plot_analysis_mean_history_%d' %
                             (sampler.number_of_adaptations)), dpi=400)

    # pprint(analysis.std_history)
    '''

    # create covid_adaptive_test[output_column].yml file
    config_yml = {'backup_dir': work_dir_adapt,
                  'number_of_adaptations': sampler.number_of_adaptations
                  }

    with open(config_yaml_file, 'w') as f:
        yaml.dump(config_yml, f, default_flow_style=False)

    with open(os.path.join(work_dir_adapt, "log_file.txt"), 'a') as log_file:
        log_file.write('%-20s -> number_of_adaptations = %-5d  total_current_runs = %d\n' %
                       ("covid_adapt", sampler.number_of_adaptations, campaign.campaign_db._next_run))

    backup_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)


@task
def covid_look_ahead(config, output_column="cumDeath", extra_lable='', ** args):
    '''
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                       ║
    ║  fab eagle_vecma covid_look_ahead:GB_suppress,output_column=cumDeath,extra_lable=''   ║
    ║                                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    '''
    print('╔═════════════════════════════════╗')
    print('║        covid_look_ahead         ║')
    print('╚═════════════════════════════════╝')

    work_dir_adapt = os.path.join(os.path.dirname(__file__),
                                  'covid_adaptive_test')
    work_dir_adapt += '[%s]' % (output_column)
    if len(extra_lable) > 0:
        work_dir_adapt += extra_lable

    config_yaml_file = os.path.join(
        os.path.dirname(__file__),
        "covid_adaptive_test[%s]%s.yml" % (output_column, extra_lable)
    )

    load_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)

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
    analysis = uq.analysis.SCAnalysis(
        sampler=sampler, qoi_cols=[output_column])
    analysis.load_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # if we decided to step backs to runs, we need to remove some runs from
    # fetched results and also from previous generated runs by easyvvuq
    with_config(config)
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')
    remove_extra_runs(campaign.campaign_db._next_run, src, des)

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
    with hide('output', 'running'):
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
    CovidSim_ensemble(config, label=job_label, **args)

    # save campaign and sampler
    campaign.save_state(os.path.join(work_dir_adapt, "campaign_state.json"))
    sampler.save_state(os.path.join(work_dir_adapt, "campaign_sampler.pickle"))

    # create covid_adaptive_test[output_column].yml file
    config_yml = {'backup_dir': work_dir_adapt,
                  'number_of_adaptations': sampler.number_of_adaptations
                  }

    with open(config_yaml_file, 'w') as f:
        yaml.dump(config_yml, f, default_flow_style=False)

    with open(os.path.join(work_dir_adapt, "log_file.txt"), 'a') as log_file:
        log_file.write('%-20s -> number_of_adaptations = %-5d  total_current_runs = %d\n' %
                       ("covid_look_ahead", sampler.number_of_adaptations, campaign.campaign_db._next_run))

    backup_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)


@task
def covid_analyse(config, output_column="cumDeath", extra_lable='', ** args):
    '''
    ╔════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                    ║
    ║  fab eagle_vecma covid_analyse:GB_suppress,output_column=cumDeath,extra_lable=''   ║
    ║                                                                                    ║
    ╚════════════════════════════════════════════════════════════════════════════════════╝
    '''
    print('╔═════════════════════════════════╗')
    print('║         covid_analyse           ║')
    print('╚═════════════════════════════════╝')

    work_dir_adapt = os.path.join(os.path.dirname(__file__),
                                  'covid_adaptive_test')
    work_dir_adapt += '[%s]' % (output_column)
    if len(extra_lable) > 0:
        work_dir_adapt += extra_lable

    config_yaml_file = os.path.join(
        os.path.dirname(__file__),
        "covid_adaptive_test[%s]%s.yml" % (output_column, extra_lable)
    )
    load_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)

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
    # fetch results from remote machine
    job_label = campaign._campaign_dir
    job_folder_name = template(env.job_name_template + "_{}".format(job_label))

    '''
    fetch_results(regex=job_folder_name)

    # copy only output folder into local campaign_dir :)
    src = os.path.join(env.local_results, job_folder_name, 'RUNS')
    des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')

    # if we decided to step backs to runs, we need to remove some runs from
    # fetched results and also from previous generated runs by easyvvuq
    # first check local results folder
    remove_extra_runs(campaign.campaign_db._next_run, src, des)

    # rmtree(backup_dir)
    local(
        "rsync -av -m -v \
        --include='/*/' \
        --include='/*/output_dir/***'  \
        --exclude='*' \
        {}/  {} ".format(src, des)
    )
    '''
    # instead of checking job scheduler, every time_step
    # fetch results and check if we have all the required output files
    # which means all jobs are finished correctly
    time_step = 5  # minutes
    time_step_passed = 0
    while True:
        print("fetching results from remote machine ...")

        with hide('output', 'running'), settings(warn_only=True):
            fetch_results(regex=job_folder_name)

        print("Done\n")

        # copy only output folder into local campaign_dir :)
        src = os.path.join(env.local_results, job_folder_name, 'RUNS')
        des = os.path.join(work_dir_adapt, campaign._campaign_dir, 'SWEEP')

        # if we decided to step backs to runs, we need to remove some runs from
        # fetched results and also from previous generated runs by easyvvuq
        remove_extra_runs(campaign.campaign_db._next_run, src, des)

        print("Syncing output_dir ...")
        with hide('output', 'running'):
            local(
                "rsync -av -m -v \
                --include='/*/' \
                --include='/*/output_dir/***'  \
                --exclude='*' \
                {}/  {} ".format(src, des)
            )
        print("Done\n")

        unfinished_jobs = []
        for i in range(1, campaign.campaign_db._next_run):
            file_to_check = os.path.join(des, 'Run_%d' % (i), target_filename)
            if not os.path.isfile(file_to_check):
                unfinished_jobs.append(file_to_check)

        print("number of unfinished_jobs = %d" % (len(unfinished_jobs)))
        if len(unfinished_jobs) == 0:
            break
        else:
            sleep(time_step * 60)

        print("time %s minutes of 60 minutes passed " % (time_step_passed))
        time_step_passed += time_step

        if time_step_passed > 60:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('something is not right . . .')
            print('jobs are submitted in interactive queue, and some of them ')
            print('are not finished yet, check jobs')
            print('list of unfinished_jobs')
            pprint(unfinished_jobs)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    print('========================================================')
    print('All required output_files are fetched . . .')
    print('========================================================')

    campaign.collate()

    # Post-processing analysis
    analysis = uq.analysis.SCAnalysis(
        sampler=campaign._active_sampler,
        qoi_cols=[output_column]
    )

    campaign.apply_analysis(analysis)

    # save analysis state
    # this is a temporary subroutine which saves the entire state of
    # the analysis in a pickle file. The proper place for this is the database
    analysis.save_state(os.path.join(
        work_dir_adapt, "campaign_analysis.pickle"))

    # create covid_adaptive_test[output_column].yml file
    config_yml = {'backup_dir': work_dir_adapt,
                  'number_of_adaptations': sampler.number_of_adaptations
                  }

    with open(config_yaml_file, 'w') as f:
        yaml.dump(config_yml, f, default_flow_style=False)

    with open(os.path.join(work_dir_adapt, "log_file.txt"), 'a') as log_file:
        log_file.write('%-20s -> number_of_adaptations = %-5d  total_current_runs = %d\n' %
                       ("covid_analyse", sampler.number_of_adaptations, campaign.campaign_db._next_run))

    backup_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)


@task
def covid_init(config, output_column="cumDeath", extra_lable='', ** args):
    '''
    ╔════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                    ║
    ║    fab eagle_vecma covid_init:GB_suppress,output_column=cumDeath,extra_lable=''    ║
    ║                                                                                    ║
    ╚════════════════════════════════════════════════════════════════════════════════════╝
    '''
    print('╔═════════════════════════════════╗')
    print('║           covid_init            ║')
    print('╚═════════════════════════════════╝')

    work_dir_adapt = os.path.join(os.path.dirname(__file__),
                                  'covid_adaptive_test')
    work_dir_adapt += '[%s]' % (output_column)
    if len(extra_lable) > 0:
        work_dir_adapt += extra_lable

    config_yaml_file = os.path.join(
        os.path.dirname(__file__),
        "covid_adaptive_test[%s]%s.yml" % (output_column, extra_lable)
    )

    # delete work_dir is exists
    if os.path.exists(work_dir_adapt):
        rmtree(work_dir_adapt)
    os.makedirs(work_dir_adapt)

    # Set up a fresh campaign called "covid-sim-adaptive[output_column]"
    campaign_name = 'covid_sim_adaptive_%s_' % (output_column)
    campaign = uq.Campaign(name=campaign_name, work_dir=work_dir_adapt)

    # to make sure we are not overwriting the new simulation on previous ones
    job_label = campaign._campaign_dir

    # Define parameter space for the covid-sim-adaptive[output_column] app
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

    decoder = uq.decoders.SimpleCSV(target_filename=target_filename,
                                    output_columns=[output_column],
                                    header=0,
                                    delimiter='\t')

    collater = uq.collate.AggregateSamples(average=False)

    '''
    CLP1=[60 100 200 300 400]
    '''

    # Add the app
    app_name = 'covidsim-adaptive-PC_CI_HQ_SD[%s]' % (output_column)
    campaign.add_app(name=app_name,
                     params=params,
                     encoder=multiencoder_covid_sim,
                     collater=collater,
                     decoder=decoder)

    # Set the active app to be covidsim-adaptive-PC_CI_HQ_SD
    campaign.set_app(app_name)

    # parameters to vary
    vary = {
        "Proportion_of_places_remaining_open_after_closure_by_place_type_universities": cp.Uniform(0.2, 0.3),
        "Proportion_of_places_remaining_open_after_closure_by_place_type_workplaces": cp.Uniform(0.8, 1.0),
        "Residual_place_contacts_after_household_quarantine_by_place_type_elementary_school": cp.Uniform(0.2, 0.3),
        "Residual_place_contacts_after_household_quarantine_by_place_type_high_schools": cp.Uniform(0.2, 0.3),
        "Residual_place_contacts_after_household_quarantine_by_place_type_universities": cp.Uniform(0.2, 0.3),
        "Residual_place_contacts_after_household_quarantine_by_place_type_workplaces": cp.Uniform(0.2, 0.3),
        "Relative_place_contact_rate_given_social_distancing_by_place_type_elementary_school": cp.Uniform(0.8, 1.0),
        "Relative_place_contact_rate_given_social_distancing_by_place_type_high_schools": cp.Uniform(0.8, 1.0),
        "Relative_place_contact_rate_given_social_distancing_by_place_type_universities": cp.Uniform(0.6, 0.9),
        "Relative_place_contact_rate_given_social_distancing_by_place_type_workplaces": cp.Uniform(0.6, 0.9),
        "Relative_rate_of_random_contacts_if_symptomatic": cp.Uniform(0.4, 0.6),
        "Relative_level_of_place_attendance_if_symptomatic_elementary_school": cp.Uniform(0.2, 0.3),
        "Relative_level_of_place_attendance_if_symptomatic_high_schools": cp.Uniform(0.2, 0.3),
        "Relative_level_of_place_attendance_if_symptomatic_universities": cp.Uniform(0.4, 0.6),
        "Relative_level_of_place_attendance_if_symptomatic_workplaces": cp.Uniform(0.4, 0.6),
        #"CLP1": cp.Uniform(60, 400),
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

    print("sampler.number_of_adaptations = %d" %
          (sampler.number_of_adaptations))

    campaign.draw_samples()
    run_ids = campaign.populate_runs_dir()

    # copy generated run folders to SWEEP directory in config folder
    # clean SWEPP, this part should be added to FabSim3.campaign2ensemble
    path_to_config = find_config_file_path(config)
    sweep_dir = path_to_config + "/SWEEP"
    local("rm -rf %s/*" % (sweep_dir))
    # campaign2ensemble copies all run_ids, we don't need it here,
    # only new run_id generated at this step should be copied
    with hide('output', 'running'):
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

    # create covid_adaptive_test[output_column].yml file
    config_yml = {'backup_dir': work_dir_adapt,
                  'number_of_adaptations': sampler.number_of_adaptations
                  }
    with open(config_yaml_file, 'w') as f:
        yaml.dump(config_yml, f, default_flow_style=False)

    with open(os.path.join(work_dir_adapt, "log_file.txt"), 'w') as log_file:
        log_file.write('%-20s -> number_of_adaptations = %-5d  total_current_runs = %d\n' %
                       ("covid_init", sampler.number_of_adaptations, campaign.campaign_db._next_run))

    backup_campaign_files(work_dir_adapt, config_yaml_file=config_yaml_file)


def backup_campaign_files(work_dir_adapt, backup_dir=None, config_yaml_file=None):

    if backup_dir is None:
        with open(config_yaml_file, 'r') as f:
            config_yml = yaml.load(f)
        backup_dir = os.path.join(
            config_yml['backup_dir'], 'backup',
            'backup_adaptation[%d]' % (config_yml['number_of_adaptations'])
        )

    # delete backup folder
    if os.path.exists(backup_dir):
        rmtree(backup_dir)
    os.makedirs(backup_dir)

    with hide('output', 'running'):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(work_dir_adapt, backup_dir)
        )


def load_campaign_files(work_dir_adapt, backup_dir=None, config_yaml_file=None):

    if backup_dir is None:
        with open(config_yaml_file, 'r') as f:
            config_yml = yaml.load(f)
        backup_dir = os.path.join(
            config_yml['backup_dir'], 'backup',
            'backup_adaptation[%d]' % (config_yml['number_of_adaptations'])
        )

    with hide('output', 'running'):
        local(
            "rsync -av -m -v \
            --include='*.db' \
            --include='*.pickle' \
            --include='*.json' \
            --exclude='*' \
            {}/  {} ".format(backup_dir, work_dir_adapt)
        )


def remove_extra_runs(runs_number, src, des):
    # if we decided to step backs to runs, we need to remove some runs from
    # fetched results and also from previous generated runs by easyvvuq
    # first check local results folder
    curr_folders = next(os.walk(src))[1]
    valid_folders = ['Run_{:d}'.format(i)
                     for i in range(1, runs_number)]
    for folder in curr_folders:
        if folder not in valid_folders:
            rmtree(os.path.join(src, folder))
    # then check SWEPP easyvvyq folder
    curr_folders = next(os.walk(des))[1]
    for folder in curr_folders:
        if folder not in valid_folders:
            rmtree(os.path.join(des, folder))
