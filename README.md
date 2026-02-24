# Code for LEAP.



# Installation and reproducing the experiments.


For managing virtual environments we use `micromamba`, which can be installed with the command

```{sh}
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

To prepare the environment, run the following commands.

```{sh}
micromamba create -n iclr  python=3.10 # create env
micromamba activate iclr  # activate env
pip install -r requirements.txt  # setup env (install required packages)
python3 -m experiments.train_gcn --help  # list the options of the main script
```

To train the NoMP architecture on the `Letter-High` dataset, run the following command.

```{sh}
python3 -m experiments.train_gcn --dataset-name=Letter-high --model-name=NoMP --use-ect=False # run an experiment
```

Train LEAP with the following command.

```{sh}
python3 -m experiments.train_gcn --dataset-name=Letter-high --use-ect=True --learn-directions=True --use-pe=False --ect-hops=1
```
