

```
.
├── aicrowd.json                  # Submission meta information - add tags for tracks here
├── apt.txt                       # Packages to be installed inside submission environment
├── requirements.txt              # Python packages to be installed with pip
├── rollout.py                    # This will run rollouts on a batched agent
├── test_submission.py            # Run this on your machine to get an estimated score
├── run.sh                        # Submission entrypoint
├── utilities                     # Helper scripts for setting up and submission 
│   └── submit.sh                 # script for easy submission of your code
├── envs                          # Operations on the env like batching and wrappers
│   ├── batched_env.py            # Batching for multiple envs
│   └── wrappers.py   	          # Add wrappers to your env here
├── agents                        # Baseline agents for submission
│   ├── batched_agent.py          # Abstraction reference batched agents
│   ├── random_batched_agent.py	  # Batched agent that returns random actions
│   ├── rllib_batched_agent.py	  # Batched agent that runs with the rllib baseline
│   └── torchbeast_agent.py       # Batched agent that runs with the torchbeast baseline
├── nethack_baselines             # Baseline agents for submission
│    ├── other_examples  	
│    │   └── random_rollouts.py   # Barebones random agent with no batching
│    ├── rllib	                  # Baseline agent trained with rllib
│    └── torchbeast               # Baseline agent trained with IMPALA on Pytorch
└── notebooks                 
    └── NetHackTutorial.ipynb     # Tutorial on the Nethack Learning Environment

```

### How can I get going with an existing baseline?

The best current baseline is the torchbeast baseline. Follow the instructions 
[here](/nethack_baselines/torchbeast/) to install and start training 
the model (there are even some suggestions for improvements).

To then submit your saved model, simply set the `AGENT` in 
`submission config` to be `TorchBeastAgent`, and modify the 
`agent/torchbeast_agent.py` to point to your saved directory.

You can now test your saved model with `python test_submission.py`

2.  **Clone the repository**

    ```
    git clone git@gitlab.aicrowd.com:nethack/neurips-2021-the-nethack-challenge.git
    ```
    
3. **Verify you have dependencies** for the Nethack Learning Environment

    NLE requires `python>=3.5`, `cmake>=3.14` to be installed and available both when building the
    package, and at runtime.
    
    On **MacOS**, one can use `Homebrew` as follows:
    
    ``` bash
    brew install cmake
    ```
    
    On a plain **Ubuntu 18.04** distribution, `cmake` and other dependencies
    can be installed by doing:
    
    ```bash
    # Python and most build deps
    sudo apt-get install -y build-essential autoconf libtool pkg-config \
        python3-dev python3-pip python3-numpy git flex bison libbz2-dev
    
    # recent cmake version
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    sudo apt-get update && apt-get --allow-unauthenticated install -y \
        cmake \
        kitware-archive-keyring
    ```

4. **Install** competition specific dependencies!

    We advise using a conda environment for this:
    ```bash
    # Optional: Create a conda env
    conda create -n nle_challenge python=3.8 'cmake>=3.15'
    conda activate nle_challenge
    pip install -r requirements.txt
    ```
    If `pip install` fails with errors when installing NLE, please see installation requirements at https://github.com/facebookresearch/nle.


# Baselines

Although we are looking to supply this repository with more baselines throughout the first month of the competition, this repository comes with a strong IMPALA-based baseline in the directory `./nethack_baselines/torchbeast`.

Follow the instructions [here](/nethack_baselines/torchbeast/) to install and start training the model (there are even some suggestions for improvements).

The TorchBeast baseline comes with two sets of weights - the same model trained to 250 million steps, and 500 million steps. 

To download these weights, run `git lfs pull`, and check `saved_models`. 

The TorchBeast agent can then be selected by setting `AGENT=TorchBeastAgent` in the `submission_config.py`, and the weights can be changed by changing the `MODEL_DIR` in `agents/torchbeast_agent.py`. 

More information on git lfs can be found on [SUBMISSION.md](/docs/SUBMISSION.md). 
