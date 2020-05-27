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
                    elif row[0] == "InputLoadFiles":
                        InputLoadFiles = str(row[1])
    else:
        print("Error !!!  can not open CovidSimFileLocation.csv file !")
        exit()

    SimSettings = {}
    if os.path.isfile('SimSettings.csv'):
        with open('SimSettings.csv', newline='') as csvfile:
            values = csv.reader(csvfile)
            for row in values:
                if len(row) > 0:  # skip empty lines in csv
                    if row[0][0] == "#":
                        pass
                    else:
                        SimSettings[row[0]] = row[1]

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

    wpop_file_root = "eur"

    wpop_file = os.path.join(
        InputLoadFiles,
        "wpop_{0}.txt".format(wpop_file_root))
    wpop_bin = os.path.join(
        InputLoadFiles,
        "{0}_pop_density.bin".format(args.country))

    # Configure pre-parameter file.  This file doesn't change between runs:
    pp_file = os.path.join(args.paramdir, "preUK_R0=2.0.txt")
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

    # Some command_line settings
    pprint(SimSettings)
    if SimSettings.__contains__('R0'):
        r = float(SimSettings['R0'])
    else:
        r = 3.0

    rs = r / 2

    # This is the temporary network that represents initial state of the
    # simulation
    '''
    network_bin = os.path.join(
        args.outputdir,
        "Network_{0}_T{1}_R{2}.bin".format(args.country, args.threads, r))
    '''
    network_bin = os.path.join(InputLoadFiles, "Network_United_Kingdom.bin")

    # Run the no intervention sim.  This also does some extra setup which is one
    # off for each R.
    print("No intervention: {0} NoInt {1}".format(args.country, r))
    cmd = [
        exe,
        "/c:{0}".format(args.threads),
        "/A:" + admin_file
    ]

    cmd.extend([
        "/PP:" + pp_file,  # Preparam file
        "/P:" + no_int_file,  # Param file
        "/O:" + os.path.join(args.outputdir,
                             "{0}_NoInt_R0={1}".format(args.country, r)),  # Output
        "/D:" + wpop_bin,  # Where to save binary pop density
        "/L:" + network_bin,  # Where to save binary net setup
        "/R:{0}".format(rs),
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
            "/A:" + admin_file
        ]

        cmd.extend([
            "/PP:" + pp_file,
            "/P:" + cf,
            "/O:" + os.path.join(args.outputdir,
                                 "{0}_{1}_R0={2}".format(args.country, root, r)),
            "/D:" + wpop_bin,  # Binary pop density file (speedup)
            "/L:" + network_bin,  # Network to load
            "/R:{0}".format(rs),
            "98798150",
            "729101",
            "17389101",
            "4797132"
        ])
        print("Command line: " + " ".join(cmd))
        process = subprocess.run(cmd, check=True)
