#!/usr/bin/env python3
"""Run the sample data.

See README.md in this directory for more information.
"""

import argparse
import gzip
import multiprocessing
import os
import csv
import shutil
import subprocess
import sys
from pprint import pprint


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

    if os.path.isfile('CovidSimFileLocation.csv'):
        with open('CovidSimFileLocation.csv', newline='') as csvfile:
            values = csv.reader(csvfile)
            for row in values:
                if len(row) > 0:  # skip empty lines in csv
                    if row[0][0] == "#":
                        pass
                    elif row[0] == "covidsim":
                        exe = str(row[1])
                    elif row[0] == "network_bin":
                        network_bin = str(row[1])
                    elif row[0] == "wpop_bin":
                        wpop_bin = str(row[1])
    else:
        print("Error !!!  can not open CovidSimFileLocation.csv file !")
        exit()

    # Ensure output directory exists
    os.makedirs(args.outputdir, exist_ok=True)

    # The admin file to use
    admin_file = os.path.join(args.datadir, "admin_units",
                              "{0}_admin.txt".format(args.country))

    if not os.path.exists(admin_file):
        print("Unable to find admin file for country: {0}".format(
            args.country))
        print("Data directory: {0}".format(args.datadir))
        print("Looked for: {0}".format(admin_file))
        exit(1)

    # Population density file in gziped form, text file, and binary file as
    # processed by CovidSim

    # Configure pre-parameter file.  This file doesn't change between runs:
    pp_file = os.path.join(args.paramdir, "preGB_R0=2.0.txt")
    if not os.path.exists(pp_file):
        print("Unable to find pre-parameter file")
        print("Param directory: {0}".format(args.paramdir))
        print("Looked for: {0}".format(pp_file))
        exit(1)

    # Configure No intervention parameter file.  This is run first
    # and provides a baseline
    no_int_file = os.path.join(args.paramdir, "p_NoInt.txt")
    if not os.path.exists(no_int_file):
        print("Unable to find parameter file")
        print("Param directory: {0}".format(args.paramdir))
        print("Looked for: {0}".format(no_int_file))
        exit(1)

    # Configure an intervention (controls) parameter file.
    # In reality you will run CovidSim many times with different parameter
    # controls.
    control_roots = ["PC_CI_HQ_SD"]
    for root in control_roots:
        cf = os.path.join(args.paramdir, "p_{0}.txt".format(root))
        if not os.path.exists(cf):
            print("Unable to find parameter file")
            print("Param directory: {0}".format(args.paramdir))
            print("Looked for: {0}".format(cf))
            exit(1)

    r = 2.4
    rs = r / 2

    # calculate CLP1-5
    x = 60  # Off_trigger_as_proportion_of_on_trigger="60 100 200 300 400"
    z = 0.25  # On_trigger="0.25 0.5 0.75"
    q = 1000
    y = x * z

    # Run the no intervention sim.  This also does some extra setup which is one
    # off for each R.
    print("No intervention: {0} NoInt {1}".format(args.country, r))
    cmd = [
        exe,
        "/c:{0}".format(args.threads),
    ]

    cmd.extend([
        "/PP:" + pp_file,  # Preparam file
        "/A:" + admin_file,
        "/P:" + no_int_file,  # Param file
        "/O:" + os.path.join(args.outputdir,
                             "{0}_NoInt_R0={1}".format(args.country, r)),  # Output
        "/D:" + wpop_bin,  # Binary pop density file (speedup)
        "/L:" + network_bin,  # Network to load
        "/R:{0}".format(rs),
        "/CLP1:" + "100000",  # FIXED
        "/CLP2:" + "0",  # FIXED
        "98798150",  # These four numbers are RNG seeds
        "729101",
        "17389101",
        "4797132"
    ])
    print("Command line: " + " ".join(cmd))
    process = subprocess.run(cmd, check=True)

    for root in control_roots:
        cf = os.path.join(args.paramdir, "p_{0}.txt".format(root))
        print("Intervention: {0} {1} {2}".format(args.country, root, r))
        cmd = [
            exe,
            "/c:{0}".format(args.threads),
        ]

    cmd.extend([
        "/PP:" + pp_file,
        "/A:" + admin_file,
        "/P:" + cf,
        "/O:" + os.path.join(args.outputdir,
                             "{0}_{1}_R0={2}".format(args.country, root, r)),
        "/D:" + wpop_bin,  # Binary pop density file (speedup)
        "/L:" + network_bin,  # Network to load
        "/R:{0}".format(rs),
        "/CLP1:%f" % (x),
        "/CLP2:%f" % (q),
        "/CLP3:%f" % (q),
        "/CLP4:%f" % (q),
        "/CLP5:%f" % (y),
        "98798150",
        "729101",
        "17389101",
        "4797132"
    ])
    print("Command line: " + " ".join(cmd))
    process = subprocess.run(cmd, check=True)
