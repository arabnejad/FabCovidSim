import sys
import numpy as np
import json

params = {}

def find_type(name):
    fname = 'CovidSim.cpp'
    type_ = 0
    with open(fname, 'r') as f:
        for line in f:
            if '"'+name in line:
                for i,char in enumerate(line):
                    if char == '%':
                        break
                if line[i+1] == 'i':
                    type_ = 'integer'
                    break
                elif line[i+1:i+3] == 'lf':
                    type_ = 'float'
                    break
                else:
                    raise Exception(line)
    if not type_:
        print('WARNING: variable not read by covid-sim:', name)
    return type_

def add_param(var_name, covidsim_name, default):
    type_ = find_type(covidsim_name)

    # if type not found, default to float??
    if type_ == 0: type_ = 'float'

    try:
        if type_ == 'integer': default = int(default)
        elif type_ == 'float': default = float(default)
    except ValueError:
        #typecasting error, assume the parameter is a string, and write as is 
        #Occurs for instance for:
        # [Trigger incidence per cell for place closure]
        # #1
        type_ = 'string'
        print('Cannot typecast %s = %s, assuming string' % (var_name, default))
    params[var_name] = {}
    params[var_name]['default'] = default
    params[var_name]['type'] = type_

def make_template(param_file):
    with open(param_file, 'r') as inf:
        with open('template_'+param_file, 'w') as outf:
            for line in inf:
                outf.write(line)

                if line[0] == '[':
                    # found new variable
                    # var_name is string between []
                    covidsim_name = ''
                    for i in range(1,len(line)):
                        if line[i] == ']':
                            break
                        covidsim_name += line[i]
    
                    var_name = covidsim_name.replace(" ", "_")
                    var_name = var_name.replace("(", "_")
                    var_name = var_name.replace(")", "_")
                    var_name = var_name.replace("/", "_")
                    var_name = var_name.replace("-", "_")
                    var_name = var_name.replace(":", "_")


                    var_line = inf.readline().split()
                    if len(var_line) == 1:
                        outf.write('{{ '+ var_name + ' }}\n')
                        add_param(var_name, covidsim_name, var_line[0])
                    elif len(var_line) > 1:
                        ### Put exceptions here
                        if var_name == 'Proportion_symptomatic_by_age_group':
                            outf.write('{% for value in Proportion_symptomatic %}{{ value }} {% endfor %}')

                        elif var_name == 'CriticalToDeath_icdf':
                            outf.write('{% for value in mortality_curve %}{{ value }} {% endfor %}')

                        else:
                            for i in range(len(var_line)):
                                add_param(var_name+str(i), covidsim_name, var_line[i])
                                outf.write('{{ ' + var_name + str(i) + ' }} ')
                        outf.write('\n')

param_files = sys.argv[1:] # e.g. p_NoInt.txt preUK_R=2.txt
for param_file in param_files:
    make_template(param_file)
json.dump(params, open('params.json','w'))