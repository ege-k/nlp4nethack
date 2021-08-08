This is the repository for the project of the course Computational Semantics for Natural Language Processing at ETH Spring Semester 2021.

Our project is named CARL: Corpus-Augmented Reinforcement Learning by

Batuhan Tomekce, Ege Karaismailoglu, Harish Rajagopal, Johannes Dollinger

In order to reproduce our results first the environment needs to be set up.

1. Install NLE from [here](https://github.com/facebookresearch/nle)
2. Install torchbeast [here](https://github.com/facebookresearch/torchbeast)
3. Clone this repo and install the dependencies in [here](requirements.txt)
4. Now you can run [polyhydra](/nethack_baselines/torchbeast/polyhydra.py) with `bsub -W 23:50 -n 20 -R "rusage[ngpus_excl_p=8]" -R "select[gpu_model0==GeForceGTX1080Ti]" python polyhydra.py`
5. You can change the subtask and parameters from the [config](/nethack_baselines/torchbeast/config.yaml)

```

├── nethack_baselines             # Baseline agents for submission
│    ├── other_examples  	
│    ├── rllib	                  # Baseline agent trained with rllib
│    └── torchbeast               # Baseline agent trained with IMPALA on Pytorch
│    │   └── polyhydra.py         # The code to run experiments
│    │   └── config.yaml          # File to change the hyperparameters and the environment to train

```
