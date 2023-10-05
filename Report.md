# CITS3001 Project

Completed by Mathew Cook (23222623) and Simon Eason (23389488).

The two agents created as part of this project are a simple rule-based agent, and a more complex Proximal Policy Optimisation (PPO) agent.

## Analysis

### Agent 1

This agent is based upon a simple set of rules that determines what actions to take at a given time. A series of templates are loaded into memory, containing the different tiles that appear in the game. Using OpenCV, the program determines which of these tile sets appear on the screen and where using the custom ``detect_objects()`` and ``detect_all_objects()`` functions, returning the found entities to the agent.

Once the location of the Mario character has been found, the area that the agent searches in (Denoted the "Region of Interest") is narrowed down to only consider the entities within a short distance of Mario, thus greatly reducing the amount of processing that needs to take place as the entire window no longer needs to be searched.

Once the location data for obstacles and Mario has been obtained, it's passed to the ruleset function ``rule_based_action()``, which then determines which action to take. This occurs for every step of the game, and the rules can be broken into two main ones:

1.   If there is an obstacle (e.g. Gooma, Koopa, pipe, step, or gap) jump over it once a certain distance away.
2.   If haven't moved after set amount of time, then most likely stuck. Move left slightly then jump right as high as possible.

The strengths of this agent are that it is relatively easy to implement (The most time consuming part being tuning values) and that it performs very consistently given a deterministic starting state.

However, the weaknesses of this approach make it less than ideal for all scenarios. It requires tuning of the specific values used in the rules to jump at the necessary times, and as a result is highly overfitted to the first level. Especially with the rule to get the agent unstuck, the values used here are tuned for locations where it was observed the agent would consistently get stuck. As development progressed, these settings had to be constantly changed to ensure it continued to reach the end.

The agent can complete the first level, and it does it consistently, but as a result of this overfitting, it is unable to complete more than that and cannot generalise to unseen levels that we have not programmed it to complete.

### Agent 2

The second agent was implemented using Proximal Policy Optimisation (PPO) from the `stablebaselines3` library. ChatGPT describes PPO as:

>   A reinforcement learning algorithm used in machine learning to train agents where they interact with the environment to learn optimal policies. It optimises the policy in a way that ensures small updates, preventing significant deviations from the current policy, and thus promoting more stable learning. PPO has been widely adopted for training agents in various applications, including robotics and game playing.

This algorithm has been trivially implemented for this agent, taking the environment and performing pre-processing steps, then passing it to a PPO class and allowing it to learn. Pre-processing was used to speed up the learning process, and includes:

1.   Grey-scaling the environment - Reduces the amount of data passed to the algorithm by removing colour channels.
1.   Vectorising the environment - Converts the environment to one the algorithm can understand.
1.   Stacking environment frames - Gives the algorithm motion data about how things change over time.

As the algorithm runs and learns, we have configured it to save the model at set intervals to allow for viewing of the agent's progress over time. Using Nvidia's CUDA GPU platform, the agent was trained on our computers using the graphics card.

The primary strength of a PPO agent is is adaptability. Given any (suitably setup) environment and a respective reward function, and the agent will be able to develop a strategy to maximise this reward. They are much more suited to generalising rules than a rule-based agent as they don't rely on a small and hard-defined set of rules, rather using a policy that is designed to be extrapolated from to allow for dealing with unseen situations (i.e. Ones noy encountered during training).

Additionally, transfer learning is big strength of a PPO agent. During this project, an agent was trained on the `v0` environment, with training stopping after 4.5m timesteps. Another agent was trained on the more simplistic `v3` environment, and only required 800k timesteps to reach (subjectively) the same level of competence. Admittedly, neither agent was capable of completing the first level repeatedly, however they were able to semi-consistently reach the same point.

The weakness of a PPO approach is the training time. Even with a decently powerful computer and GPU, training still took multiple hours for the agent to reach some level of competence. From anecdotal observation, it took the PPO agent approximately 250k steps to complete the first level when training in the `gym-super-mario-bros-v3` environment. When running on a computer with an Nvidia GTX 1070ti, it took 45 minutes for the algorithm to be trained to this level. In comparison to the amount of time it took to tune the rule-based agent, this isn't actually very long, however the PPO agent only completed the first level once with that level of training, and was still very inconsistent with its performance (i.e. How far to the right it got), often dying upon encountering the first obstacle. Many hours, sometimes days of training are required to get the PPO agent to a consistently performing state.

## Performance metrics

### Agent 1

In terms of measuring performance, the rule-based agent is pretty simple. Given a consistent/predetermined starting state and the current tuned values, the agent will always complete the first level. However, as previously stated, these values are set specifically for this first level, and would require tweaking to work on subsequent levels.

The number of iterations/length of time it would take to adjust these values to work on different levels is solely dependent of the speed of the programmer. It would be possible to come up with a database of values, one set per level, then load the respective values for a level when that level is progressed to. This is a way this type of agent could be generalised to work across the whole game, rather than just single levels. 

Additionally, the types of rules used and how they are implemented are flexible. The rule-based agent created for this project measures the distance from obstacles, then jumps at the appropriate time to avoid them. Different rules could be created to, for example, kill as many enemies as possible and get the highest score. This could even be achieved with the current approach by increasing the distance the agent jumps, so that is lands on the enemies' head.

### Agent 2

The PPO agent can be described as a "black box"; data is fed in, and an action is returned. It is very hard to see the thought process of the agent to determine what action is being taken. The agent uses the environment's reward function to determine how it can change its policy to maximise the reward, which can lead to unexpected outcomes. 

For example, the training for this agent was left to run overnight, finishing with >8m timesteps in the morning. When this model file was loaded and run, the agent repeatedly ran into the first obstacle and died, before respawning and doing the same thing again. It never took any other actions, only running as fast as it could into the obstacle.

We believe the agent is trying to maximise the reward it obtains is as short a time possible, and that it determined that dying repeatedly was the best way of doing that.

As such, a subjective/observation approach was used to pick the model that could consistently make it the furthest (From the models saved at set intervals during training). Models from later in the training cycle all appeared to be converging to this unexplained state, so earlier ones were primarily selected.

## Visualisation and debugging

### Agent 1

The rule-based agent uses custom optical template recognition functions that return the location of desired templates, namely the location of the Mario player character, the location of moving enemies (e.g. Goombas, Koopas) and the location of static obstacles (e.g. Pipes, gaps, etc.). This collection of location data is passed to the function `draw_borders_on_detected_objects()`, which draws coloured borders around the templates that have been detected in the scene. An example of this is shown below:

![Borders on detected objects](report_assets/rule_based_debug.png)

By restricting the borders to only be drawn in the area the agent is looking for obstacles, it is possible to see what the agent sees, and thus fine tune the rules that define how early the agent should jump to clear the obstacle. By restricting the area the agent looks for obstacles, it removes the need to check the entire screen, and thus greatly improves the framerate the environment runs at. However, a "failsafe" was implemented, so if the algorithm lost sight of the Mario character at any point in that restricted region, then it would search the entire view until the character was located, updating the area afterwards to include this new position.

### Agent 2

Developing and debugging the PPO agent was significantly more difficult. As stated earlier, the PPO algorithm acts as a sort of "black box", with us having very little understanding about the decisions it was making. There are several "knobs and dials" that can be adjusted for this algorithm to produce different results, namely `learning_rate`, `n_steps` and the environment used.

In an attempt to prevent converging to the strange "run at the first enemy and die" state, the learning rate was adjusted up and down, with seemingly little change occurring after hours of training. We attempted to train using the `v3` environment, but the same issue occurred. However, using this environment, then running the model in the `v0` one did show the agent was able to transfer its learning (As previously stated).

What also made this agent more difficult to develop was the time it took for a useable agent to be trained. A change would be made to the PPO algorithm's parameters, but we would have to wait hours to see the result of this change. The `tensorboard` tool allowed us to view the logs of the training process, showing things such as `loss` and `explained variance`, but this was of little help when attempting to pick values to use as parameters.

Additionally, the window that appeared during training showed the agent showed the agent progressing well, however the agent in the respective model file did not perform as well.

Overall, a trial-and-error approach was used to develop this agent.

