# CITS3001 Project

Completed by Mathew Cook (23222623) and Simon Eason (23389488).

The two agents created as part of this project are a simple rule-based agent, and a more complex [INSERT TYPE HERE] agent.

## Analysis

### Agent 1

This agent is based upon a simple set of rules that determines what actions to take at a given time. A series of templates are loaded into memory, containing the different tiles that appear in the game. Using OpenCV, the program determines which of these tile sets appear on the screen and where using the ``detect_objects()`` and ``detect_all_objects()`` functions, returning the found entities to the agent.

Once the location of the Mario character has been found, the area that the agent searches in (Denoted the "Region of Interest") is narrowed down to only consider the entities within a short distance of Mario, thus greatly reducing the amount of processing that needs to take place as the entire window no longer needs to be searched.

Once the location data for obstacles and Mario has been obtained, it's passed to the ruleset function ``rule_based_action()``, which then determines which action to take. This occurs for every step of the game, and the rules can be broken into two main ones:

1.   If there is an obstacle (e.g. Gooma, Koopa, pipe, step, or gap) jump over it once a certain distance away.
2.   If haven't moved after set amount of time, then most likely stuck. Move left slightly then jump right as high as possible.

The strengths of this agent are that it is relatively easy to implement (The most time consuming part being tuning values) and that it performs very consistently given a deterministic starting state.

However, the weaknesses of this approach make it less than ideal for all scenarios. It requires tuning of the specific values used in the rules to jump at the necessary times, and as a result is highly overfitted to the first level. Especially with the rule to get the agent unstuck, the values used here are tuned for locations where it was observed the agent would consistently get stuck. As development progressed, these settings had to be constantly changed to ensure it continued to reach the end.

The agent can complete the first level, and it does it consistently, but as a result of this overfitting, it is unable to complete more than that.



### Agent 2



## Performance metrics



## Visualisation and debugging

