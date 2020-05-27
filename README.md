



# FabCovidsim
This is a FabSim3 /EasyVVUQ plugin for Covid-19 simulation


## Dependencies:

[FabSim3](https://github.com/djgroen/FabSim3.git) : `git clone https://github.com/djgroen/FabSim3.git`

[COVID-19 CovidSim microsimulation model developed Imperial College, London](https://github.com/mrc-ide/covid-sim)


## Installation
Simply type `fab localhost install_plugin:FabCovidsim` anywhere inside your FabSim3 install directory.

### FabSim3 Configuration
Once you have installed the required dependencies, you will need to take a few small configuration steps:
1. Go to `(FabSim Home)/deploy`
2. Open `machines_user.yml`
3. for `qcg` machine, add these lines
``` yaml
qcg:
  ...
  ...
  # setting for Imperial College COVID-19 simulator
  cores: 28
  budget: "vecma2020"
  # job wall time for each job, format P[nD]T[nH][nM]
  # nD - number of days, nH - number of hours, nM - number of minutes
  job_wall_time : "PT20M" # job wall time for each single job without PJ
  PJ_size : "2" # number of requested nodes for PJ
  PJ_wall_time : "PT50M" # job wall time for PJ
  modules:
    loaded: ["python/3.7.3", "r/3.6.1-gcc620"] # do not change
    unloaded: [] #
```
4. for `eagle_vecma` machine, add these lines
``` yaml
eagle_vecma:
  ...
  ...
  # setting for Imperial College COVID-19 simulator
  cores: 28
  budget: "vecma2020"
  # job wall time for each job, format Days-Hours:Minutes:Seconds
  job_wall_time : "0-0:20:00" # job wall time for each single job without PJ
  PJ_size : "2" # number of requested nodes for PJ
  PJ_wall_time : "0-00:50:00" # job wall time for PJ
  modules:
    loaded: ["python/3.7.3", "r/3.6.1-gcc620"] # do not change
    unloaded: [] #
```    
   <br/> _NOTE: you can changes the values of these attributes, but do not change the modules load list._
  
## Job submission via command line
1. To run a single job, simply type:
  >``` sh
  > fab <qcg/eagle_vecma> CovidSim:UK_sample[,memory=MemorySize][,label=your_lable]
  > ```   
  > _NOTE:_
  >   - by default **memory=20GB** .
  >   

2. To run the ensemble, you can type, simply type:
  >``` sh
  > fab <qcg/eagle_vecma> CovidSim_ensemble:UK_sample[,<memory=MemorySize>][,replicas=replica_number]
  > ```   
  > _NOTE:_
  >   -  **replicas=N** : will generate N replicas
  >    - if you want to run multiple simulations with different configuration, to do that, create your own folder name under `SWEEP` directory, and change the parameters files, there are some examples under `/config_files/SWEEP_examples` folder,
  >    - if you want to use QCG-PJ with `eagle_vecma` make sure that you first install it by `fab eagle_vecma install_app:QCG-PilotJob,virtual_env=True`
  >
  > _Examples:_
  >   -  `fab eagle_vecma CovidSim_ensemble:UK_sample`
  >   -  `fab qcg CovidSim_ensemble:UK_sample,PilotJob=True`
  >   -  `fab qcg CovidSim_ensemble:UK_sample,replicas=5,PilotJob=True`
  >   -  `fab eagle_vecma CovidSim_ensemble:UK_sample,PilotJob=True`
  >   -  `fab eagle_vecma CovidSim_ensemble:UK_sample,replicas=5`

## Running a standard EasyVVUQ - CovidSim campaign via Python

NOTE: As is stands now, everything below needs EasyVVUQ functionality present in the `dev` branch, so check out this branch first.

This demonstrates how to use a standard EasyVVUQ campaign on the CovidSim code. By 'standard' we mean non-dimension adaptive, where each input parameter is sampled equally. There are two Python scripts that need to be executed:

1. `standard_covid_easyvvuq/covid_init_SC.py`: a standard EasyVVUQ campaign up to and including job submission.
2. `standard_covid_easyvvuq/covid_analyse_SC.py`: job retrieval and post processing.

The `FabCovidSim` plugin is called from within `standard_covid_easyvvuq/covid_init_SC.py` via
``` python
import fabsim3_cmd_api as fab
fab.run_uq_ensemble(config, campaign.campaign_dir, script="CovidSim",
                    machine="eagle_vecma", PilotJob=True)
```
Here, `config` is the name of the config file directory that is used for the code, which are located in `config_files/`. Furthermore `script="CovidSim"` refers to `/templates/CovidSim`, which contains the instructions to execute a single job.

To retrieve the results from the remote machine, the following is executed in `standard_covid_easyvvuq/covid_analyse_SC.py`:
``` python
fab.get_uq_samples(config, campaign.campaign_dir, sampler._number_of_samples,
                   machine='eagle_vecma')
```

## Running a dimension-adaptive EasyVVUQ - CovidSim campaign via Python

The Python scripts for a dimension-adaptive sampling plan are located in `adaptive_covid_easyvvuq/` and `dummy_covid_easyvuq`. The former will run CovidSim, the latter will take the same input files but instead of calling CovidSim, it'll run a simple analytic model. The tutorial is demonstrated on the dummy model, which you can just quickly run on your localhost to get a feel for the dimension adaptive sampler. To run the dummy model, add the following line to `machines_user.yml`:

```
localhost:
  covid_dummy_exec: "<your fab home dir>/plugins/FabCovidsim/config_files/dummy_covid/run_sample.py"
```

The dummy model varies:
```python
#parameters to vary
vary = {
    "Relative_household_contact_rate_after_closure": cp.Uniform(1.5*0.8, 1.5*1.2),
    "Relative_spatial_contact_rate_after_closure": cp.Uniform(1.25*0.8, 1.25*1.2),
    "Relative_household_contact_rate_after_quarantine": cp.Uniform(1.5*0.8, 1.5*1.2),
    "Residual_spatial_contacts_after_household_quarantine": cp.Uniform(0.25*0.8, 0.25*1.2),
    "Household_level_compliance_with_quarantine": cp.Uniform(0.5, 0.9),
    "Individual_level_compliance_with_quarantine": cp.Uniform(0.9, 1.0),
    "Proportion_of_detected_cases_isolated":cp.Uniform(0.85, 0.95),
    "Residual_contacts_after_case_isolation":cp.Uniform(0.25*0.8, 0.25*1.2),
    "Relative_household_contact_rate_given_social_distancing":cp.Uniform(1.1, 1.25*1.2),
    "Relative_spatial_contact_rate_given_social_distancing":cp.Uniform(0.05, 0.15)
}
```

These same parameters are hard coded in the analytic dummy model, so if you select different ones, remember to also change `run_sample.py`. In any case it doesn't matter much, since the analytic model is set up in such a way that the first parameter is the most important, then the second, then the third etc. If working properly, the dimension-adaptive sampler should pick up on this sensitivity, and place more samples along the first couple of directions, while ignoring the last parameters after the initial samples.

To place these initial samples, execute the first two files, which are almost the same as for a standard EasyVVUQ campaign:

1. `dummy_covid_easyvvuq/dummy_init.py`: an almost standard EasyVVUQ campaign up to and including job submission.
2. `dummy_covid_easyvvuq/dummy_analyse.py`: job retrieval and post processing.

The main difference is that in `dummy_init.py` the sampler must be chosen as

```python
sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                quadrature_rule="C",
                                sparse=True, growth=True,
                                midpoint_level1=True,
                                dimension_adaptive=True)
```
Here:

* `polynomial_order=1`: do not change, will be adaptively increased for influential parameters. (Technically, it'll change the quadrature order for different (combinations of) parameters)
* `quadrature_rule="C":`selects the Clenshaw Curtis quadrature rule. This not required, although it is common.
* `sparse = True`: selects a sparse grid. This is required.
* `growth = True`: selects a nested quadrature rule (a quadrature rule such that a 1D rule of order p contains all points of the same rule of order p-1). Also not required, but is efficient in high dimensions. Note that this can only be selected with a subset of all quadrature rules in Chaospy, including Clenshaw Curtis.
* `midpoint_level1=True`: this means that the first iteration of the dimension-adaptive sampler consists of a single sample. Keep this fixed for now, but this might change later. Perhaps we could start the dimension-adaptivity from an existing sampling plan.
* `dimension_adaptive=True`: selects the dimension-adaptive sparse grid sampler (opposed to the isotropic sparse grid sampler, which treats each input the same).

The other difference with respect to a standard EasyVVUQ campaign, is the need to save the complete state of the sampler and analysis class. In an adaptive setting we need information from the previous iteration to determine where to go next. Right now EasyVVUQ does not do this, it only saves part of the sampler to the database. As a temporary hacky solution, we can now save their state to a `pickle` file via:

```python
sampler.save_state("states/covid_sampler_state.pickle")
analysis.save_state("states/covid_analysis_state.pickle")
```

To recap, just execute `dummy_init.py` once, followed by `dummy_analyse.py`, which is also executed once. The latter retrieves the samples and stores the analysis object. 

Next, the following 2 files are executed, multiple times and one after the other:

1. `dummy_covid_easyvvuq/dummy_look_ahead.py`: Takes the current configuration of points, computes so-called adimissble
points in the `look_forward` subroutine. These points are used in the following adaptation step to decide along which dimension to place more samples.
2. `dummy_covid_easyvvuq/dummy_adapt.py`: Takes the admissible points computed in the `look_ahead` step to decide along 
which dimension to place more samples. Update the configuration of points, and return to `look_ahead`step.

The code for the `look_ahead` step is given by:
```python
#load campaign, sampler and analysis object
state_file = 'states/covid_easyvvuq_state.json'
campaign = uq.Campaign(state_file=state_file, work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
sampler.load_state("states/covid_sampler_state.pickle")
campaign.set_sampler(sampler)
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
analysis.load_state("states/covid_analysis_state.pickle")

#required parameter in the case of a Fabsim run
skip = sampler.count

#look-ahead step (compute the code at admissible forward points)
sampler.look_ahead(analysis.l_norm)

#proceed as usual
campaign.draw_samples()
campaign.populate_runs_dir()

#save campaign and sampler
campaign.save_state("states/covid_easyvvuq_state.json")
sampler.save_state("states/covid_sampler_state.pickle")

#run the UQ ensemble at the admissible forward points
#skip (int) = the number of previous samples: required to avoid recomputing
#already computed samples from a previous iteration
fab.run_uq_ensemble(config, campaign.campaign_dir, script='Dummy_CovidSim',
                    machine="localhost", skip=skip)
```
**Basically, all this does is 1) load everything (campaign, sampler, analysis); 2) call `sampler.look_ahead`; 3) store everything again; 4) run ensemble.**

Some parameters need a bit more explaining. The `skip` parameter is used to prevent recomputing already computed samples. We set it equal to the number of samples which are already computed through `skip=sampler.count`. This is required because `fab.run_uq_ensemble` calls the `campaign2ensemble` function of Fabsim. This copies the EasyVVUQ runs from `campaign.campaign_dir` to the Fabsim config sweep directory: 

```python
<your fabsim home dir>/plugins/FabCovidsim/config_files/dummy_covid/SWEEP
```

All runs in this directory will be executed. Without modification, we would therefore be executing old runs from the 2nd time we execute `dummy_look_ahead.py` onward. By setting `skip=sampler.count`, we first delete all run directories `SWEEP/Run_X` for `X <= skip` before we run the ensemble, therefore only executing new samples. If there's a better way of doing so let me know.

The main function call here is `sampler.look_ahead(analysis.l_norm)`. First, `l_norm` is a set of multi indices, denoting the order of the 1D quadrature (and therefore the number of points) used per input parameter. For instance:
``` python
l_norm = array([[1, 1]])
```
means we have 2 input parameters, both of which have a 1D quadrature rule of order 1. Lets say the chosen quadrature rule generates `x = [0.5]` as point for order 1. Then the 2D grid is built as a tensor product `[0.5] x [0.5] = [0.5, 0.5]`. Hence `l_norm = array([[1, 1]])` means we have a 2D grid consisting of just one point `[0,5, 0.5]`. If on the other hand we have
``` python
l_norm = array([[1, 1],
                [1, 2],
                [2, 1]])
```
we are building a grid using a linear combination of 3 tensor products. If our 1D quadrature rule generates `x = [0.0, 0.5, 1.0]` for order 2, our grid consistis of the points:
```python 
l_norm = [1, 1]  ==> [0.5] x [0.5] = [[0.5, 0.5]]
```
and
```python
l_norm = [1, 2]  ==> [0.5] x [0.0, 0.5, 1.0] = [[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]]
```
and
```python
l_norm = [2, 1]  ==> [0.0, 0.5, 1.0] x [0.5] = [[0.0, 0.5], [0.5, 0.5], [1.0, 0.5]]
```
Notice that this makes a "cross" of 5 (unique) points in the input space. A standard EasyVVUQ campaign always has a single `l_norm`, such as for instance `[2,2]`, which leads to a full "square" of 9 points: `[0.0, 0.5, 1.0] x [0.0, 0.5, 1.0]`. Sparse grids on the other hand, build the grid from the bottom up, using a linear combination of `l_norm` multi indices, which can lead to more sparse sampling plans. 

The array `analysis.l_norm` contains the currently accepted multi indices, and therefore the current configuration of points. What `analysis.look_ahead(analysis.l_norm)` does is take these multi indices, and compute new "candidate" `l_norm` entries, which are stored in `sampler.admissible_idx`. These new candidate multi indices will (probably) contain unsampled points, which are sampled using `fab.run_uq_ensemble`.

Then, `dummy_adapt.py` is executed, and its main task is to select one of the candidate multi indices in `sampler.admissible_idx`, which will be added to `analysis.l_norm`. The main script looks like:

```python
state_file = 'states/covid_easyvvuq_state.json'
campaign = uq.Campaign(state_file=state_file, work_dir=work_dir)
print('========================================================')
print('Reloaded campaign', campaign.campaign_dir.split('/')[-1])
print('========================================================')
sampler = campaign.get_active_sampler()
sampler.load_state("states/covid_sampler_state.pickle")
campaign.set_sampler(sampler)
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=output_columns)
analysis.load_state("states/covid_analysis_state.pickle")

fab.get_uq_samples(config, campaign.campaign_dir, sampler._number_of_samples,
                   machine='localhost')
campaign.collate()

#compute the error at all admissible points, select direction with
#highest error and add that direction to the grid
data_frame = campaign.get_collation_result()
analysis.adapt_dimension(output_columns[0], data_frame)

#save everything
campaign.save_state("states/covid_easyvvuq_state.json")
sampler.save_state("states/covid_sampler_state.pickle")
analysis.save_state("states/covid_analysis_state.pickle")
```

**In a nutshell, this 1) loads everything (campaign, sampler, analysis); 2) retrieves the new samples from the remote; 3) call `analysis.adapt_dimension` 4) stores everything.**

The function `analysis.adapt_dimension` takes the name of the quantity of interest (QoI), and the sample database as input. For every multi index in `sampler.admissible_idx`, it computes the so-called "hierarchical surplus", which is the difference between the new sample of our QoI (`output_columns[0]`), and the polynomial approximation of that sample at the previous iteration. In code this reads as
```python
for xi in admissible_idx:
  #find the location of the current xi in the global grid
  idx = np.where((xi == self.sampler.xi_d).all(axis=1))[0][0]
  #hierarchical surplus error at xi
  hier_surplus = samples[idx] - self.surrogate(qoi, xi)
```
The surplus is therefore used as a local error estimator, and the multi index in `sampler.admissible_idx` with the highest surplus will get added to `analysis.l_norm`, and then `dummy_look_ahead.py` can get executed again. This goes round and round until you are happy with the result or until you run out of computational budget. None of this is automated (yet). 

**Note**: if you want to do post processing, do it in `dummy_adapt.py`.

**Note 2**: the Sobol indices may become inaccurate. I think this is due to imbalance in sampling density, i.e. the fact that certain dimensions get very little samples, while other get a lot. I might be able to fix this by interpolation, but this is on the todo list. In the mean time, there is another means of retrieving sensitivity information, see below.

To extract sensitivity information, we can just look at which multi indices, and by extension which parameters, get selected for refinement. By calling `analysis.adaptation_histogram()` we simply plot a histogram of the maximum quadrature order per parameter (the max of `analysis.l_norm` per column). This gives

No refinement yet (after `dummy_init.py` and `dummy_analyse.py`)

![alt text](https://github.com/arabnejad/FabCovidsim/blob/dev/dummy_covid_easyvvuq/figs/adapt0.png)

First refinement step

![alt text](https://github.com/arabnejad/FabCovidsim/blob/dev/dummy_covid_easyvvuq/figs/adapt1.png)

Second refinement step

![alt text](https://github.com/arabnejad/FabCovidsim/blob/dev/dummy_covid_easyvvuq/figs/adapt2.png)

Third refinement step

![alt text](https://github.com/arabnejad/FabCovidsim/blob/dev/dummy_covid_easyvvuq/figs/adapt3.png)

Fourth refinement step

![alt text](https://github.com/arabnejad/FabCovidsim/blob/dev/dummy_covid_easyvvuq/figs/adapt4.png)

Remember that the Dummy model wa created such that the order of importance was the same as the order of parameters in the `vary` dict. The figures above show that the dimension adaptive samples does pick up on this. Note however that from the 3rd to the 4th refinement, there seems to be no change. However, if we examine `analysis.l_norm`:

```python
array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
       [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]])
```
we see that the last selected multi index was `[2, 2, 1, 1, 1, 1, 1, 1, 1, 1]`. This is a *joint* (interaction) refinement between the first and the second input. This histogram does not show such interaction refinements, and should be interpreted as some type of first-order sensitivity index.
