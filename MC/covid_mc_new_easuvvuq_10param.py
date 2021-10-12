"""
==============================================================================
QMC COVIDSIM CAMPAIGN IN A SINGLE SCRIPT
==============================================================================
"""
import os
import json
import easyvvuq as uq
import chaospy as cp
# import numpy as np
# import matplotlib.pyplot as plt
import fabsim3_cmd_api as fab
from custom_new import CustomEncoder

home = os.path.abspath(os.path.dirname(__file__))
output_columns = ["cumDeath"]
WORK_DIR = '/home/wouter/VECMA/Campaigns/tmp'
# WORK_DIR = '/tmp'
# FabSim3 config name 
CONFIG = 'PC_CI_HQ_SD_DAS_10_param'
# Simulation identifier
ID = '_DAS_10'
# EasyVVUQ campaign name
CAMPAIGN_NAME = CONFIG + ID
# name and relative location of the outpu file name
TARGET_FILENAME = 'output_dir/United_Kingdom_PC_CI_HQ_SD_R0=2.6.avNE.severity.xls'
# location of the EasyVVUQ database
DB_LOCATION = "sqlite:///" + WORK_DIR + "/campaign%s.db" % ID
# Use QCG PiltJob or not
PILOT_JOB = True

#set to True if starting a new campaign
INIT = True
if INIT:

    # Define parameter space
    params = json.load(open(home + '/../templates_campaign_full1/params.json'))

    #manually add mortality factor, which parameterizes CriticalToDeath_icdf
    #and SARIToDeath_icdf
    params["Mortality_factor"] = {"default":1, "type": "float"}

    #manually add Proportion_symptomatic to seed the array with a uniform value
    params["Proportion_symptomatic"] = {"default": 0.66, "type": "float"}

    #manually add Relative_spatial_contact_rates_by_age_power, used to parameterize
    #Relative_spatial_contact_rates_by_age_array
    params["Relative_spatial_contact_rates_by_age_power"] = {"default": 1, "type": "float"}

    # Modify the random seeds manually
    # params['Random_seeds0']['default'] = 98798250
    # params['Random_seeds1']['default'] = 729201
    # params['Random_seeds2']['default'] = 17389301
    # params['Random_seeds3']['default'] = 4797332

    #overwrite the type of these 4 variables to avoid sampling a delay of 1.1232 dys for instance
    params['Delay_to_start_case_isolation']['type'] = 'integer'
    params['Delay_to_start_household_quarantine']['type'] = 'integer'
    params['Duration_of_case_isolation']['type'] = 'integer'
    params['Length_of_time_households_are_quarantined']['type'] = 'integer'

    # Create an encoder and decoder
    directory_tree = {'param_files': None}

    multiencoder_p_PC_CI_HQ_SD = uq.encoders.MultiEncoder(
        uq.encoders.DirectoryBuilder(tree=directory_tree),
        uq.encoders.JinjaEncoder(
            template_fname=home + '/../templates_campaign_full1/p_PC_CI_HQ_SD.txt',
            target_filename='param_files/p_PC_CI_HQ_SD.txt'),
        CustomEncoder(
            template_fname=home + '/../templates_campaign_full1/preGB_R0=2.0.txt',
            target_filename='param_files/preGB_R0=2.0.txt'),
        uq.encoders.JinjaEncoder(
            template_fname=home + '/../templates_campaign_full1/p_seeds.txt',
            target_filename='param_files/p_seeds.txt')
    )

    ###########################
    # Set up a fresh campaign #
    ###########################

    actions = uq.actions.Actions(
        uq.actions.CreateRunDirectory(root=WORK_DIR, flatten=True),
        uq.actions.Encode(multiencoder_p_PC_CI_HQ_SD),
    )

    campaign = uq.Campaign(
        name=CAMPAIGN_NAME,
        db_location=DB_LOCATION,
        work_dir=WORK_DIR,
        verify_all_runs=False
    )

    campaign.add_app(
        name=CAMPAIGN_NAME,
        params=params,
        actions=actions
    )

    #########################
    # parameters to vary    #
    # place types           #
    # ----------------------#
    # 0 = elementary school #
    # 1 = high school       #
    # 2 = university        #
    # 3 = workplaces        #
    #########################

    vary = {
        ###########################
        # Intervention parameters #
        ###########################
        # "Relative_household_contact_rate_after_closure": cp.Uniform(1.5*0.8, 1.5*1.2),
        # "Relative_spatial_contact_rate_after_closure": cp.Uniform(1.25*0.8, 1.25*1.2),
        # "Relative_household_contact_rate_after_quarantine": cp.Uniform(2.0*0.8, 2.0*1.2),
        # "Residual_spatial_contacts_after_household_quarantine": cp.Uniform(0.25*0.8, 0.25*1.2),
        "Household_level_compliance_with_quarantine": cp.Uniform(0.5, 0.9),
        # "Individual_level_compliance_with_quarantine": cp.Uniform(0.9, 1.0),
        # "Proportion_of_detected_cases_isolated":cp.Uniform(0.6, 0.8),
        # "Residual_contacts_after_case_isolation":cp.Uniform(0.25*0.8, 0.25*1.2),
        # "Relative_household_contact_rate_given_social_distancing":cp.Uniform(1.1, 1.25*1.2),
        "Relative_spatial_contact_rate_given_social_distancing":cp.Uniform(0.15, 0.35),
        "Delay_to_start_household_quarantine":cp.DiscreteUniform(1, 3),
        # "Length_of_time_households_are_quarantined":cp.DiscreteUniform(12, 16),
        "Delay_to_start_case_isolation":cp.DiscreteUniform(1, 3),
        # "Duration_of_case_isolation":cp.DiscreteUniform(5, 9),
        ######################
        # Disease parameters #
        ######################
        "Symptomatic_infectiousness_relative_to_asymptomatic": cp.Uniform(1,2),
        "Proportion_symptomatic": cp.Uniform(0.4,0.8),
        "Latent_period": cp.Uniform(3,6),
        # "Mortality_factor": cp.Uniform(0.8,1.2),
        # "Reproduction_number": cp.Uniform(2,2.7),
        # "Infectious_period": cp.Uniform(11.5, 15.6),
        # "Household_attack_rate": cp.Uniform(0.1, 0.19),
        # "Household_transmission_denominator_power": cp.Uniform(0.7, 0.9),
        "Delay_from_end_of_latent_period_to_start_of_symptoms": cp.Uniform(0, 1.5),
        # "Relative_transmission_rates_for_place_types0": cp.Uniform(0.08, 0.15),
        # "Relative_transmission_rates_for_place_types1": cp.Uniform(0.08, 0.15),
        # "Relative_transmission_rates_for_place_types2": cp.Uniform(0.05, 0.1),
        # "Relative_transmission_rates_for_place_types3": cp.Uniform(0.05, 0.07),
        # "Relative_spatial_contact_rates_by_age_power": cp.Uniform(0.25, 4),
        ######################
        # Spatial parameters #
        ######################
        # "Proportion_of_places_remaining_open_after_closure_by_place_type2": cp.Uniform(0.2, 0.3),
        # "Proportion_of_places_remaining_open_after_closure_by_place_type3": cp.Uniform(0.8, 1.0),
        # "Residual_place_contacts_after_household_quarantine_by_place_type0": cp.Uniform(0.2, 0.3),
        # "Residual_place_contacts_after_household_quarantine_by_place_type1": cp.Uniform(0.2, 0.3),
        # "Residual_place_contacts_after_household_quarantine_by_place_type2": cp.Uniform(0.2, 0.3),
        # "Residual_place_contacts_after_household_quarantine_by_place_type3": cp.Uniform(0.2, 0.3),
        # "Relative_place_contact_rate_given_social_distancing_by_place_type0": cp.Uniform(0.8, 1.0),
        # "Relative_place_contact_rate_given_social_distancing_by_place_type1": cp.Uniform(0.8, 1.0),
        # "Relative_place_contact_rate_given_social_distancing_by_place_type2": cp.Uniform(0.6, 0.9),
        "Relative_place_contact_rate_given_social_distancing_by_place_type3": cp.Uniform(0.6, 0.9),
        # "Relative_rate_of_random_contacts_if_symptomatic": cp.Uniform(0.4, 0.6),
        # "Relative_level_of_place_attendance_if_symptomatic0": cp.Uniform(0.2, 0.3),
        # "Relative_level_of_place_attendance_if_symptomatic1": cp.Uniform(0.2, 0.3),
        # "Relative_level_of_place_attendance_if_symptomatic2": cp.Uniform(0.4, 0.6),
        "Relative_level_of_place_attendance_if_symptomatic3": cp.Uniform(0.4, 0.6),
        # "CLP1": cp.Uniform(60, 400)
        #############
        # Leftovers #
        #############
        # "Kernel_scale": cp.Uniform(0.9*4000, 1.1*4000),
        # "Kernel_Shape": cp.Uniform(0.8*3, 1.2*3),
        # "Kernel_shape_params_for_place_types0": cp.Uniform(0.8*3, 1.2*3),
        # "Kernel_shape_params_for_place_types1": cp.Uniform(0.8*3, 1.2*3),
        # "Kernel_shape_params_for_place_types2": cp.Uniform(0.8*3, 1.2*3),
        # "Kernel_shape_params_for_place_types3": cp.Uniform(0.8*3, 1.2*3),
        # "Kernel_scale_params_for_place_types0": cp.Uniform(0.9*4000, 1.1*4000),
        # "Kernel_scale_params_for_place_types1": cp.Uniform(0.9*4000, 1.1*4000),
        # "Kernel_scale_params_for_place_types2": cp.Uniform(0.9*4000, 1.1*4000),
        # "Kernel_scale_params_for_place_types3": cp.Uniform(0.9*4000, 1.1*4000),
        # "Param_1_of_place_group_size_distribution0": cp.DiscreteUniform(20, 30),
        # "Param_1_of_place_group_size_distribution1": cp.DiscreteUniform(20, 30),
        # "Param_1_of_place_group_size_distribution2": cp.DiscreteUniform(80, 120),
        # "Param_1_of_place_group_size_distribution3": cp.DiscreteUniform(8, 12),
        # "Proportion_of_between_group_place_links0": cp.Uniform(0.8*0.25, 1.2*0.25),
        # "Proportion_of_between_group_place_links1": cp.Uniform(0.8*0.25, 1.2*0.25),
        # "Proportion_of_between_group_place_links2": cp.Uniform(0.8*0.25, 1.2*0.25),
        # "Proportion_of_between_group_place_links3": cp.Uniform(0.8*0.25, 1.2*0.25),
    }

    # sampler = uq.sampling.RandomSampler(vary, max_num=4)
    sampler = uq.sampling.quasirandom.LHCSampler(vary, max_num=60)

    ###########################################
    # Associate the sampler with the campaign #
    ###########################################
    campaign.set_sampler(sampler)

    #########################################
    # draw all of the finite set of samples #
    #########################################
    campaign.execute().collate()

    # run the UQ ensemble
    fab.run_uq_ensemble(CONFIG, campaign.campaign_dir, script='CovidSim',
                        machine="eagle_vecma", PJ=PILOT_JOB)
else:

    ###################
    # reload Campaign #
    ###################
    campaign = uq.Campaign(name=CAMPAIGN_NAME, db_location=DB_LOCATION)
    print("===========================================")
    print("Reloaded campaign {}".format(CAMPAIGN_NAME))
    print("===========================================")

    sampler = campaign.get_active_sampler()
    campaign.set_sampler(sampler, update=True)

    # #wait for job to complete
    # fab.wait(machine="eagle_vecma")

    # #check if all output files are retrieved from the remote machine
    # all_good = fab.verify(CONFIG, campaign.campaign_dir,
    #                       TARGET_FILENAME,
    #                       machine="eagle_vecma", PJ=PILOT_JOB)

    # if all_good:
    #     #copy the results from the FabSim results dir to the
    #     fab.get_uq_samples(CONFIG, campaign.campaign_dir, sampler.max_num,
    #                         machine='eagle_vecma')
    # else:
    #     print("Not all samples executed correctly")
    #     import sys; sys.exit()

    #####################
    # execute collate() #
    #####################
    decoder = uq.decoders.SimpleCSV(
        target_filename=TARGET_FILENAME,
        output_columns=output_columns, dialect='excel-tab')

    actions = uq.actions.Actions(
        uq.actions.Decode(decoder)
    )
    campaign.replace_actions(CAMPAIGN_NAME, actions)
    campaign.execute().collate()
    data_frame = campaign.get_collation_result()
