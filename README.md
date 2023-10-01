# CITS3001 Project

Completed by Mathew Cook (23222623) and Simon Eason (23389488).

## Setup

A `requirements.txt` file is provided with the necessary dependencies to install. This project was developed with `virtualenv` and `pip`. 

If a virtual environment is desired, run the following commands:

```
virtualenv venv
[ACTIVATE VIRTUAL ENVIRONMENT. STEPS WILL DIFFER BASED ON OS]
```

Once the environment has been created, then the packages can be installed. To run the training for the second agent, it is necessary to install [`PyTorch`](https://pytorch.org/get-started/locally/) prior to installing the `requirements.txt`. 

**NOTE**: CUDA (Version 11.8 was used) has to be installed to run training on the GPU.

Once installed, run the following command to download and install the remaining packages:

```
pip install -r requirements.txt
```



## Agent 1 - Rule-based

This is a simple rule-based agent that can be run with the command:

```
python rulebased.py
```

Two windows will appear, one of them the actual game being played by the agent, and the other the debug window showing what the agent sees via optical template recognition.

This agent relies on specific values set during the development process, and as such can only complete the first level.



## Agent 2 - Proximal Policy Optimisation

This agent uses `Stable Baseline`'s Proximal Policy Optimisation algorithm.

The training can be run with the command:

```
python stablebaselines.py
```

During training, the model is saved at set intervals to the `models` directory, and logs are saved to `logs`. These logs can be viewed with the command:

```
tensorboard --logdir=logs
```

To run the agent with a given model, run the command:

```
python stablebaselines.py [PATH TO MODEL ZIP]
```

