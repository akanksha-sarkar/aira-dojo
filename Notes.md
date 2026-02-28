# Notes 

1. Make sure environment variables are set!!
2. Make sure to load the correct apptainer module!!!

## Other notes

If you want to change anything for all tasks, overwrite in task/agentSSL/_default.yaml.

## How to Use

1. Set AGENTSSL_DATA_DIR="path to agentSSL directory" (in .env file or with export)
2. Run the following command:
```
python -m dojo.main_run \ +_exp=run_example \ task=agentSSL/_default \  task.name=domain1  \ task.setting=setA \  task.subset=k1 \  task.seed=seed0
```

## AgentSSL Structure (Folder in /share/j_sun)

```
agentSSL/
├── domain1/ # Ex. CUB or RESIC
│ ├── data/ # Raw training data (unannotated) (train + val split is probably easiest at this point)
│ ├── test/ # Test data and annotations
| ├── setA/ (setting)
│ │ ├── build/ # folder to build the shot and seed folders
│ │ │ ├── evaluate.py # Setting specific evaluation function
│ │ │ ├── description.md # Setting specific task description
│ | │ └── build.py # File to build the rest of the folders
│ │ ├── k1/ # 1 shot 
│ │ │ ├── seed0/ # Random seed
│ │ │ │ ├── annotations/ # Annotation files
│ │ │ │ │ ├── train.json # train labels
│ │ │ │ │ ├── unlabeled.json # index unlabeled data
│ │ │ │ ├── src/ # API & dataset setup functions
│ │ │ │ ├── evaluate.py # Evaluation function
│ | │ │ └── description.md # Task description and notes
│ │ │ ├── seed26/ # Random seed
│ │ │ └── seed42/ # Random seed
│ │ ├── k2/ # 2 shot 
│ │ ├── k5/ # 5 shot 
│ │ ├── k10/ # 10 shot 
│ │ └── k20/ # 20 shot 
| ├── setB/ (setting)
| └── setC/ (setting)
├── domain2
├── domain3
├── domain4
└── Other domains...
```

## Apptainer Environment View (What the agent sees)

```
├── /data/ # r/o (same content as agentSSL/{domain}/data/) 
├── /cache/ # r/w (not sure how this works yet, probably in sand file, probably gets overwritten each time?)
├── /pseudolabel/ # r/w 
├── /work/ # r/o (same content as agentSSL/{domain}/{setting}/{shot}/{seed})
├── /run_tmp/ # r/o (program agent writes goes to /run_tmp/program.py)
```
### Binds (Implemented in main_run.py + /cache is probably in the sand file)
- /data/ : agentSSL/{domain}/data/
- /cache/ : ???
- /pseudolabel/ : ???
- /work/ : agentSSL/{domain}/{setting}/{shot}/{seed}

## Logging (Implemented in main_run.py)

Right now to get the outputs to be in a more readable location using an environment variable $EXPERIMENT_DIR we have, 
- cfg.logger.output_dir : EXPERIMENT_DIR
- cfg.solver.checkpoint_path : os.path.join(experiment_dir, "checkpoint")
- cfg.interpreter.working_dir : os.path.join(experiment_dir, "workspace_agent")