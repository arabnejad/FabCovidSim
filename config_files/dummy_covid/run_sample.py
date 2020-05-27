#!/usr/bin/env python3
"""Run the sample data.

See README.md in this directory for more information.
"""

import argparse
# import gzip
import multiprocessing
import os
# import csv
# import shutil
# import subprocess
# import sys
# from pprint import pprint
import numpy as np

def search_string_in_file(file_name, string_to_search):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_number = 0
    list_of_results = []
    # Open the file in read only mode
    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            line_number += 1
            if string_to_search in line:
                # If yes, then add the line number & line as a tuple in the list
                list_of_results.append((line_number, line.rstrip()))
 
    # Return list of tuples containing line numbers and lines where string is found
    return list_of_results

def poly_model(theta):
    
    sol = 1.0
    for i in range(d):
        sol *= 3 * a[i] * theta[i]**2 + 1.0
    return sol/2**d

def try_remove(f):
    try:
        os.remove(f)
    except OSError as e:
        pass


def parse_args():
    """Parse the arguments.

    On exit: Returns the result of calling argparse.parse()

    args.covidsim is the name of the CovidSim executable
    args.datadir is the directory with the input data
    args.paramdir is the directory with the parameters in it
    args.outputdir is the directory where output will be stored
    args.threads is the number of threads to use
    """
    parser = argparse.ArgumentParser()
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity isn't available
        cpu_count = multiprocessing.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2

    script_path = os.path.dirname(os.path.realpath(__file__))

    # Default values
    data_dir = script_path
    param_dir = os.path.join(script_path, "param_files")
    output_dir = os.getcwd()

    parser.add_argument(
        "country",
        help="Country to run sample for")
    parser.add_argument(
        "--datadir",
        help="Directory at root of input data",
        default=script_path)
    parser.add_argument(
        "--paramdir",
        help="Directory with input parameter files",
        default=param_dir)
    parser.add_argument(
        "--outputdir",
        help="Directory to store output data",
        default=output_dir)
    parser.add_argument(
        "--threads",
        help="Number of threads to use",
        default=cpu_count
    )
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = parse_args()
    
    vary = ["Relative_household_contact_rate_after_closure",
        "Relative_spatial_contact_rate_after_closure",
        "Relative_household_contact_rate_after_quarantine",
        "Residual_spatial_contacts_after_household_quarantine",
        "Household_level_compliance_with_quarantine",
        "Individual_level_compliance_with_quarantine",
        "Proportion_of_detected_cases_isolated",
        "Residual_contacts_after_case_isolation",
        "Relative_household_contact_rate_given_social_distancing",
        "Relative_spatial_contact_rate_given_social_distancing"]


    d = len(vary)
    a = np.ones(d)
    for i in range(1, d):
        a[i] = a[i-1]/2
    
    # fname = os.path.join(args.paramdir, 'template_p_PC7_CI_HQ_SD.txt')
    fname = './param_files/template_p_PC7_CI_HQ_SD.txt'
    fp = open(fname, 'r')
    lines = fp.readlines()
    fp.close()
    
    xi = []
    
    for param in vary:
        location = search_string_in_file(fname, param.replace('_', ' '))
        xi.append(float(lines[location[0][0]]))
    
    result = poly_model(xi)

    fname = './output_dir/foo.severity.xls'

    result = poly_model(xi)

    # output csv file
    header = 'cumDeath'
    np.savetxt(fname, np.array([result]),
               delimiter=",", comments='',
               header=header)
    
