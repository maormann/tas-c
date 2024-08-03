# TAS Configurator (TAS-C)

The TAS Configurator creates **[Tele-Assistance System](https://people.cs.kuleuven.be/~danny.weyns/software/TAS/)** **[Prism models](https://www.prismmodelchecker.org/)** based on **[YAML](https://yaml.org/)** configurations using **[Jinja templates](https://jinja.palletsprojects.com/)**. The configuration is designed in such a way that it is possible to process entire batches of model variants and have PRISM or **[EvoChecker](https://www-users.york.ac.uk/~sg778/EvoChecker/)** automatically calculate properties for these models.

This repository is used to investigate the **[PARLEY paradigm of Carwehl et al.](https://arxiv.org/abs/2401.17187)** for uncertainty reduction in the domain of TAS.

## Requirements

- Git
- Docker
- [Taskfile](https://taskfile.dev/installation/) the better Makefile

## Quick Start

TAS-C has been designed to work with disposable docker containers as a virtualization environment. However, it should also be possible to use all commands without Docker. In this case, the environment variable NO_DOCKER_ENV should be passed to Task with any value or written to the .env file.

- Run `task build`
- Run `task` to see all available tasks
- Run `task --summary run-prism-dice` to see more information about a task e.g. `run-prism-dice`
- Run `x=3 task run-prism-dice` to see if prism working properly

## Tutorial

Follow this tutorial to see how TAS-C works.
If you want to see more information about what task TAS-C is currently doing, you can include LOGGING_LEVEL=debug in the .env file.

### Aggregated failure rate

To start simply, let's examine the aggregated failure rate experiment.

- Examine the file `configurations/aggregated_failure_rate.yaml`
- You will find a simple configuration with 4 different experiments. The first experiment is always the basic experiment and all subsequent experiments always extend the basic experiment. These are separated by YAML documents with `---`. This means that all subsequent experiments are variants of the basic experiment.
- Run `task run-aggregated-failure-rate`
- Navigate to the `results/aggregated_failure_rate` path
- Each execution creates a subdirectory with date and time. In this directory there is an `aggregated_failure_rate.csv`, which shows a summary of all experiments of the aggregated failure rates. The experiments directory contains all experiments with their configuration, results, generated Prism models and more.
- The `aggregated_failure_rate.csv` should have the following content and is reproducible 
  ```
  FR_010;0.00199962866548312
  FR_008;0.0005118445477992153
  FR_025;0.010988233174210025
  FR_MIX;0.0014063722556198627
  ```

### Basic

The standard configuration shows how to verify properties of the PRSIM model directly with tas-c.

- Run `task run-basic`
- Examine the file `configurations/basic.yaml` and look for the model_checking part
  ```yaml
  model_checking:
    run: true
    rounds: 15
    prism_computation_engine: "sparse"  # mtbdd, sparse, hybrid, explicit
    # constants steps and alarms will be generated
    properties:
        - property: 'R{"s1_invocations"}=? [C <= steps]'
        - property: 'R{"s2_invocations"}=? [C <= steps]'
        - property: 'R{"s3_invocations"}=? [C <= steps]'
        - property: 'R{"total_costs"}=? [C <= steps]'
        - property: 'R{"model_drift"}=? [C <= steps]'
        - property: 'R{"time_per_alarm"}=? [C <= steps] / alarms'

  ```
- All results of the model checking can be found in the file `results/basic/*/model_checking.csv` in the order the properties are defined
  ```
  basic;4.442117447982671;1.6374985337173837;1.0964459249597456;21.471851971835704;2.9190278705853454;1.2779856129360714
  ```
