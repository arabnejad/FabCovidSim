#!/bin/bash

# ./run_covid_adaptive_test_cumDeath.sh 2>&1 | tee execution_log.txt 
extra_lable="no_CLP1"
output_column="cumDeath"
END=50



fabsim eagle_vecma covid_init:GB_suppress,output_column=$output_column,extra_lable=$extra_lable

fabsim eagle_vecma covid_analyse:GB_suppress,output_column=$output_column,extra_lable=$extra_lable

for i in $(seq 1 $END)
do
	fabsim eagle_vecma covid_look_ahead:GB_suppress,output_column=$output_column,extra_lable=$extra_lable
    fabsim eagle_vecma covid_adapt:GB_suppress,output_column=$output_column,extra_lable=$extra_lable
done


